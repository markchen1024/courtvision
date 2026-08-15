"""SAM2 tracking exactly as the tutorial does it, for a like-for-like comparison.

track_sam2.py is not the notebook's tracker, and the differences were only
visible once the notebook was read line by line:

  detector    the notebook prompts SAM2 with basketball-player-detection-3
              -ycjdo/4 -- the same model it uses everywhere else. Ours took its
              prompts from the dense cache in track_bytetrack.py, which calls
              project.py's detector: koppolusameer/rfdetr-basketball-player-ball
              -referee-detection, a different model off HuggingFace. On frame 0
              of the 30:05 segment the notebook's detector returns 11 players
              and ours returns 10.
  classes     PLAYER_CLASS_IDS = [3, 4, 5, 6, 7] -- player, player-in-possession,
              player-jump-shot, player-layup-dunk, player-shot-block. Ours kept
              only class_name == "player". They agree on frame 0 here, but a
              frame with someone mid-shot would not.
  thresholds  confidence 0.4, iou 0.9.
  weights     sam2.1_hiera_large through segment-anything-2-real-time's camera
              predictor. Ours ran sam2.1_b through the ultralytics wrapper.

    python pipeline/track_sam2_tutorial.py --video web/media/det_final.mp4 \
        --out out/tracks_sam2tut_detfinal.json

Output matches track_sam2.py's schema, so identify.py and render_final.py take
either one and the two can be scored against each other.

--reprompt-seconds re-detects and re-seeds mid-clip, which fixes tracking and
breaks naming. It is off by default. Measured on seg_01m10.87s_19s at 3.0s:

                        prompt once      re-seed every 3s
    merge lasts         7.3s, to the end   3.0s
    Brunson             lost from 11.9s    re-acquired at 15.0s
    Duren duplicate     ids 9 and 10       gone (prompts are de-duplicated)
    names               10, all correct    11, one of them invented

The invention is the point. Harris ended up on two ids -- 5 until 15s and 12
from 18s -- because a re-seed can hand a player a fresh id. The roster
matching in identify.py assigns one number per track, so with #12 taken by the
stronger of the two, the other was pushed onto the only number left and became
`#7 Paul Reed`, who was not on the floor. Brunson's recovered track went
unnamed for a related reason: its crops span two different people, so the team
clustering split 11-10 and put it on the wrong side of a roster that has no
11.

Both need the same missing piece, tracklet association: tracks that never
coexist and read the same number are one player and must be merged before the
assignment. Until that exists, a wrong name on screen is worse than a missing
one, so the default stays 0.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import config
import oncourt
from check_lineup import dedupe
from progress import Progress

SAM2_REPO = Path(r"C:\Users\fqche\segment-anything-2-real-time")
DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
CONFIDENCE = 0.4          # notebook cell 25
IOU_THRESHOLD = 0.9       # notebook cell 25
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]   # notebook cell 31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint",
                    default=str(SAM2_REPO / "checkpoints" / "sam2.1_hiera_large.pt"))
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the clip backwards, prompting on the last frame. "
                         "SAM2's memory is causal, so this is genuinely "
                         "different evidence rather than the same run twice -- "
                         "a player it loses into an occlusion going forward may "
                         "be held coming the other way. Rows keep the video's "
                         "own frame numbers so the two sidecars can be fused.")
    ap.add_argument("--prompt-at", type=int,
                    help="frame to prompt from; the run then walks away from it "
                         "in whichever direction --reverse selects. The last "
                         "frame is often a bad choice going backwards: on "
                         "seg_01m10.87s_19s it held a courtside spectator who "
                         "projects 0.04m inside the sideline, so he passed the "
                         "court test, took a prompt slot, and Brunson was never "
                         "tracked at all.")
    ap.add_argument("--prompt-margin", type=float, default=oncourt.MARGIN_M,
                    help="metres past the lines a prompt may stand. 3 is right "
                         "for frame 0, where being generous costs nothing; a "
                         "NEGATIVE value demands the feet be that far INSIDE "
                         "the court, which is how a man sitting on the line "
                         "gets excluded without touching real players.")
    ap.add_argument("--no-court-filter", action="store_true",
                    help="prompt with every frame-0 detection, the notebook's "
                         "behaviour, including anyone in the crowd")
    ap.add_argument("--reprompt-seconds", type=float, default=0.0,
                    help="re-detect and re-seed every N seconds, inheriting "
                         "track ids. 3.0 is the measured setting; 0, the "
                         "default, prompts once on frame 0 as the notebook "
                         "does and lets a merge last to the final frame. Off "
                         "by default because it currently costs more than it "
                         "buys -- see the tracker's docstring")
    ap.add_argument("--stitch-iou", type=float, default=0.2,
                    help="IoU at which a fresh detection is the same object as "
                         "an existing track")
    ap.add_argument("--stitch-px", type=float, default=250.0,
                    help="fallback: how far a track may have moved from its "
                         "last box and still be the same player, when no box "
                         "overlaps it. This is what recovers a merge -- the "
                         "man who was lost is nowhere near where his track was "
                         "left sitting.")
    args = ap.parse_args()

    config.load_env()
    config.inference_env()   # key, cache path, GPU provider -- before the import

    import cv2
    import torch
    import supervision as sv
    from inference import get_model

    # hydra resolves the config path relative to the sam2 package root
    os.chdir(SAM2_REPO)
    from sam2.build_sam import build_sam2_camera_predictor

    predictor = build_sam2_camera_predictor(args.sam2_config, args.checkpoint)
    detector = get_model(model_id=DETECTION_MODEL_ID)

    video = Path(args.video)
    if not video.is_absolute():
        video = Path(__file__).resolve().parent.parent / video
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Which frame the run starts from, and which way it walks. SAM2's memory is
    # causal, so a backward pass is not the same run played twice: at a merge
    # it arrives with a different history, and it prompts from a frame where
    # the players it will meet are still apart. Rows are still written under
    # the video's own frame numbers, so a forward and a backward sidecar line
    # up and can be fused.
    start = args.prompt_at if args.prompt_at is not None else (
        total - 1 if args.reverse else 0)
    start = max(0, min(start, total - 1))
    order = (list(range(start, -1, -1)) if args.reverse
             else list(range(start, total)))

    def read_at(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return fr if ok else None

    frame = read_at(order[0])
    if frame is None:
        raise SystemExit("empty video")
    print(f"{'reverse' if args.reverse else 'forward'} pass: prompting on "
          f"frame {order[0]} ({order[0] / fps:.2f}s), "
          f"{len(order)} frames to {order[-1]}")

    kp_model = None if args.no_court_filter else oncourt.keypoint_model()

    def detect(frame, announce=False):
        """Players standing on the floor, one box each."""
        result = detector.infer(frame, confidence=CONFIDENCE,
                                iou_threshold=IOU_THRESHOLD)[0]
        det = sv.Detections.from_inference(result)
        det = det[np.isin(det.class_id, PLAYER_CLASS_IDS)]
        # DEVIATION from the notebook, which prompts with every player-class
        # detection on frame 0. On seg_02m27.00s_14s one of them was a
        # spectator in a CUNNINGHAM #2 jersey in the front row: it took a
        # prompt, was tracked for the whole clip, had '2' read off its back
        # twenty times and was given a name, while the real Cade Cunningham was
        # never tracked at all. Anyone whose feet are not on the floor is not a
        # player. The test falls open when the court cannot be solved, so a
        # frame without landmarks still prompts with everything.
        if kp_model is not None and len(det):
            on, _, note = oncourt.feet_on_court(frame, det.xyxy, model=kp_model,
                                                margin_m=args.prompt_margin)
            if announce:
                print(f"court check: {note}")
            det = det[on]
        # DEVIATION: the notebook prompts with every surviving box. Two boxes on
        # one man then become two tracks that never separate -- Duren carried
        # ids 9 and 10 through the whole of seg_01m10.87s_19s. Re-seeding every
        # few seconds would mint a fresh pair each time, so the duplicates have
        # to go at the source. 0.7, not 0.5: players guard each other body to
        # body and real boxes overlap a lot.
        if len(det):
            det = det[dedupe(det.xyxy.tolist(), det.confidence.tolist())]
        return det

    detections = detect(frame, announce=True)
    detections.tracker_id = np.arange(1, len(detections.class_id) + 1)
    if len(detections) == 0:
        raise SystemExit("no players on frame 0 to prompt with")
    print(f"prompting SAM2 with {len(detections)} players from frame 0 "
          f"(classes {sorted(set(int(c) for c in detections.class_id))})")

    # SAM2Tracker.prompt_first_frame, notebook cell 40
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.load_first_frame(frame)
        for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
            predictor.add_new_prompt(frame_idx=0, obj_id=int(obj_id),
                                     bbox=np.asarray([xyxy], dtype=np.float32))

    def stitch(previous, boxes):
        """Give each fresh box the id of the track it continues.

        One-to-one, and that is the point. When two tracks have merged onto one
        player, only one of them can be given that player's box; the other is
        forced onto whatever is left, which is the man it lost. A greedy
        nearest-box rule would hand both to the same detection and the merge
        would survive the re-seed.

        Overlap decides it where there is any. Where there is none -- exactly
        the case of a track sitting on the wrong man while its own player
        stands elsewhere -- distance decides it, up to --stitch-px.
        """
        from scipy.optimize import linear_sum_assignment

        tids = sorted(previous)
        assigned, taken = [None] * len(boxes), set()
        if tids and len(boxes):
            prev = np.array([previous[t] for t in tids], np.float32)
            cur = np.array(boxes, np.float32)
            # IoU first
            iou = np.zeros((len(tids), len(cur)))
            for i, a in enumerate(prev):
                for j, b in enumerate(cur):
                    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                    inter = ix * iy
                    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
                             - inter)
                    iou[i, j] = inter / union if union > 0 else 0.0
            rows, cols = linear_sum_assignment(-iou)
            spare_tids, spare_cols = [], set(range(len(cur)))
            for i, j in zip(rows, cols):
                if iou[i, j] >= args.stitch_iou:
                    assigned[j] = tids[i]
                    taken.add(tids[i])
                    spare_cols.discard(j)
                else:
                    spare_tids.append(i)
            spare_tids += [i for i in range(len(tids)) if i not in rows]
            # then distance, for the tracks left holding nothing
            if spare_tids and spare_cols:
                cols_left = sorted(spare_cols)
                centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                cost = np.array([[np.hypot(*(np.subtract(centre(prev[i]),
                                                         centre(cur[j]))))
                                  for j in cols_left] for i in spare_tids])
                for i, j in zip(*linear_sum_assignment(cost)):
                    if cost[i, j] <= args.stitch_px:
                        assigned[cols_left[j]] = tids[spare_tids[i]]
                        taken.add(tids[spare_tids[i]])
        return assigned, taken

    tracks = {}
    idx, prev_idx, step = order[0], None, 0
    next_id = int(detections.tracker_id.max()) + 1
    reprompt_every = (int(round(fps * args.reprompt_seconds))
                      if args.reprompt_seconds > 0 else 0)
    if reprompt_every:
        print(f"re-seeding every {reprompt_every} frames "
              f"({args.reprompt_seconds}s)")
    reseeds = []
    # The direction is part of the job's identity, not a detail inside it. Both
    # passes writing out/progress/sam2-tutorial.json meant the second clobbered
    # the first's live file, and in the archive the two runs this stage exists
    # to fuse arrived as two rows with the same name, told apart only by their
    # timestamps. "sam2" is still the matched prefix, so the stage is unchanged.
    prog = Progress("sam2-tutorial-reverse" if args.reverse else "sam2-tutorial",
                    total=total or None, video=args.video, artifact=args.out,
                    meta={"direction": "reverse" if args.reverse else "forward",
                          "prompt_frame": order[0],
                          "reprompt_s": args.reprompt_seconds})
    while True:
        if reprompt_every and step and step % reprompt_every == 0:
            # DEVIATION from the notebook, which prompts once and lets SAM2's
            # appearance memory carry the rest of the clip. It cannot recover
            # from a merge: on seg_01m10.87s_19s, Brunson and Beasley made
            # contact at 11.9s and both tracks walked away on Beasley for the
            # remaining seven seconds. The detector sees both men on every one
            # of those frames. So ask it again, every few seconds, and hand the
            # answers back to the tracks that own them.
            fresh = detect(frame)
            previous = {r["tid"]: r["box"] for r in tracks.get(prev_idx, [])}
            assigned, taken = stitch(previous, fresh.xyxy.tolist())
            born = []
            for j, tid in enumerate(assigned):
                if tid is None:
                    assigned[j] = next_id
                    born.append(next_id)
                    next_id += 1
            lost = sorted(set(previous) - taken)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.load_first_frame(frame)
                for box, obj_id in zip(fresh.xyxy, assigned):
                    predictor.add_new_prompt(
                        frame_idx=0, obj_id=int(obj_id),
                        bbox=np.asarray([box], dtype=np.float32))
            reseeds.append({"frame": idx, "t": round(idx / fps, 2),
                            "detections": len(fresh), "new": born,
                            "dropped": lost})
            if born or lost:
                print(f"  re-seed at {idx/fps:5.2f}s: {len(fresh)} boxes"
                      + (f", new ids {born}" if born else "")
                      + (f", dropped {lost}" if lost else ""), flush=True)

        # SAM2Tracker.propagate, notebook cell 40
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = predictor.track(frame)
        ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        masks = np.array([
            sv.filter_segments_by_distance(m, relative_distance=0.03, mode="edge")
            for m in masks
        ])
        xyxy = sv.mask_to_xyxy(masks=masks)

        rows = []
        for tid, box in zip(ids, xyxy):
            if box[2] > box[0] and box[3] > box[1]:   # a lost object yields an empty mask
                rows.append({"tid": int(tid), "box": [float(v) for v in box]})
        tracks[idx] = rows

        prev_idx, step = idx, step + 1
        # `at` is where in the clip this frame sits, which on a reverse pass
        # runs the opposite way to the step count. The watcher's still follows
        # `at`; without it a backward run showed the far end of the footage.
        prog.step(note=f"frame {idx}, {len(rows)} objects",
                  at=(idx / (total - 1)) if total > 1 else 0.0)
        if step % 200 == 0:
            print(f"  {step} frames...", flush=True)
        if step >= len(order):
            break
        idx = order[step]
        frame = read_at(idx)
        if frame is None:
            break
    cap.release()

    lifetimes = {}
    for i, rows in tracks.items():
        for r in rows:
            a, b = lifetimes.get(r["tid"], (i, i))
            lifetimes[r["tid"]] = (min(a, i), max(b, i))
    prog.done(note=f"{step} frames, {len(lifetimes)} objects")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent.parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "video": args.video, "every": 1, "fps": fps,
        "tracker": f"segment-anything-2-real-time camera predictor "
                   f"({Path(args.checkpoint).name})",
        "detector": DETECTION_MODEL_ID,
        "prompted": len(detections),
        "reprompt_seconds": args.reprompt_seconds,
        "reverse": bool(args.reverse),
        "reseeds": reseeds,
        "frames": {str(k): v for k, v in tracks.items()},
    }))
    spans = [(b - a) / fps for a, b in lifetimes.values()]
    print(f"wrote {out}: {step} frames, {len(lifetimes)} of {len(detections)} "
          f"prompted objects ever seen")
    if spans:
        print(f"  lifetime median {sorted(spans)[len(spans)//2]:.1f}s  "
              f"max {max(spans):.1f}s (clip is {step/fps:.0f}s)")


if __name__ == "__main__":
    main()
