"""The notebook's pipeline, run as one script. Cells 45 through 72, in order.

Nothing here is our design. Where this repo has its own version of a step, the
notebook's is used instead, and the deviation is noted:

  teams        TeamClassifier fit on 0.4-scaled crops sampled every 30 frames
               (cell 45), predicted ONCE on frame 0, and frozen by a
               ConsecutiveValueTracker(n_consecutive=1). identify.py instead
               votes over crops sampled through the clip.
  cluster->club
               the notebook maps K-means cluster 0/1 to a club by hand -- cell
               53 has the wrong order commented out above the right one, because
               the cluster index is arbitrary. --team0/--team1 is that hand
               step; --show-clusters writes the two crop grids to look at first.
               identify.py instead infers the mapping from roster overlap.
  numbers      RF-DETR number regions every 5th frame, matched to players by
               MASK IoS >= 0.9 (cell 71 uses sv.mask_iou_batch with
               OverlapMetric.IOS), read by SmolVLM2, confirmed by
               ConsecutiveValueTracker(n_consecutive=3). identify.py uses a
               majority vote and can match on boxes.
  render       MaskAnnotator + RichLabelAnnotator with "#<number> <surname>"
               from the roster (cell 72).

SAM2 runs twice: once to collect readings, once to render, because the
validators only hold their final answer after the whole clip has been seen and
keeping 1049 frames of masks in memory is not an option outside Colab.

    python pipeline/run_tutorial.py --video web/media/det_final.mp4 \
        --show-clusters out/clusters      # look, then decide the order
    python pipeline/run_tutorial.py --video web/media/det_final.mp4 \
        --team0 pistons --team1 knicks --out out/tutorial_result.mp4
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import config
from progress import Progress

SAM2_REPO = Path(r"C:\Users\fqche\segment-anything-2-real-time")
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"   # cell 25
PLAYER_CONFIDENCE = 0.4
PLAYER_IOU_THRESHOLD = 0.9
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]                                    # cell 31
NUMBER_CLASS_ID = 2                                                   # cell 29
OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
OCR_PROMPT = "Read the number."
STRIDE = 30                                                           # cell 45
NUMBER_EVERY = 5                                                      # cell 71
IOS_THRESHOLD = 0.9                                                   # cell 71
CROP_SCALE = 0.4                                                      # cell 45


def detect_players(model, frame, sv, class_agnostic_nms=False):
    kwargs = dict(confidence=PLAYER_CONFIDENCE, iou_threshold=PLAYER_IOU_THRESHOLD)
    if class_agnostic_nms:
        kwargs["class_agnostic_nms"] = True
    result = model.infer(frame, **kwargs)[0]
    det = sv.Detections.from_inference(result)
    return det[np.isin(det.class_id, PLAYER_CLASS_IDS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--rosters", default="web/data/rosters_det.json")
    ap.add_argument("--team0", help="club name for K-means cluster 0")
    ap.add_argument("--team1", help="club name for K-means cluster 1")
    ap.add_argument("--show-clusters", metavar="DIR",
                    help="write both cluster's crops and stop, so the order "
                         "can be decided by eye (the notebook's cell 49)")
    ap.add_argument("--out", default="out/tutorial_result.mp4")
    ap.add_argument("--out-json", default="out/tutorial_identities.json")
    ap.add_argument("--checkpoint",
                    default=str(SAM2_REPO / "checkpoints" / "sam2.1_hiera_large.pt"))
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent

    def path(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    config.load_env()
    config.inference_env()

    import cv2
    import torch
    import supervision as sv
    from inference import get_model
    from sports import ConsecutiveValueTracker, TeamClassifier

    video = str(path(args.video))
    detector = get_model(model_id=PLAYER_DETECTION_MODEL_ID)

    # ---- cell 45: crops for the team classifier, every STRIDE frames --------
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crops, idx = [], 0
    prog = Progress("tutorial-crops", total=total // STRIDE)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STRIDE == 0:
            det = detect_players(detector, frame, sv, class_agnostic_nms=True)
            for box in sv.scale_boxes(xyxy=det.xyxy, factor=CROP_SCALE):
                crop = sv.crop_image(frame, box)
                if crop.size:
                    crops.append(crop)
            prog.step(note=f"frame {idx}, {len(crops)} crops")
        idx += 1
    cap.release()
    prog.done(note=f"{len(crops)} crops")
    print(f"collected {len(crops)} crops from every {STRIDE}th frame")

    # ---- cell 48: fit ------------------------------------------------------
    print("fitting TeamClassifier (SigLIP + UMAP + K-means)...")
    team_classifier = TeamClassifier(
        device="cuda" if torch.cuda.is_available() else "cpu")
    team_classifier.fit(crops)

    if args.show_clusters:
        # cell 49: the notebook looks at both clusters and decides which is which
        out_dir = path(args.show_clusters)
        out_dir.mkdir(parents=True, exist_ok=True)
        teams = np.array(team_classifier.predict(crops))
        for t in (0, 1):
            picks = [c for c, k in zip(crops, teams) if k == t][:40]
            if not picks:
                continue
            h = max(c.shape[0] for c in picks)
            w = max(c.shape[1] for c in picks)
            cols = 10
            rows = (len(picks) + cols - 1) // cols
            sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
            for i, c in enumerate(picks):
                r, col = divmod(i, cols)
                sheet[r*h:r*h + c.shape[0], col*w:col*w + c.shape[1]] = c
            cv2.imwrite(str(out_dir / f"cluster_{t}.jpg"), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote cluster_0.jpg and cluster_1.jpg to {out_dir}")
        print("look at them, then rerun with --team0 <club> --team1 <club>")
        return

    if not (args.team0 and args.team1):
        raise SystemExit(
            "--team0 and --team1 are required (the notebook hand-writes this "
            "mapping in cell 53). Run --show-clusters first.")

    rosters = json.loads(path(args.rosters).read_text(encoding="utf-8"))
    club_by_cluster = {0: args.team0, 1: args.team1}
    names_by_club = {c: {str(p["num"]): p["name"] for p in rosters[c]}
                     for c in rosters if isinstance(rosters[c], list)}
    for club in club_by_cluster.values():
        if club not in names_by_club:
            raise SystemExit(f"{club!r} is not in {args.rosters} "
                             f"({', '.join(names_by_club)})")

    ocr = get_model(model_id=OCR_MODEL_ID)

    os.chdir(SAM2_REPO)
    from sam2.build_sam import build_sam2_camera_predictor

    number_validator = ConsecutiveValueTracker(n_consecutive=3)   # cell 71
    team_validator = ConsecutiveValueTracker(n_consecutive=1)     # cell 71

    def prompt_and_track(render_pass, sink=None, annotators=None):
        """One pass of cells 40/41/71 over the clip."""
        predictor = build_sam2_camera_predictor(args.sam2_config, args.checkpoint)
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("empty video")

        det = detect_players(detector, frame, sv)
        det.tracker_id = np.arange(1, len(det.class_id) + 1)
        if not render_pass:
            boxes = sv.scale_boxes(xyxy=det.xyxy, factor=CROP_SCALE)
            first_crops = [sv.crop_image(frame, b) for b in boxes]
            teams0 = np.array(team_classifier.predict(first_crops))
            team_validator.update(tracker_ids=det.tracker_id.tolist(),
                                  values=teams0.tolist())
            print(f"prompting SAM2 with {len(det)} players from frame 0")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(det.xyxy, det.tracker_id):
                predictor.add_new_prompt(frame_idx=0, obj_id=int(obj_id),
                                         bbox=np.asarray([xyxy], dtype=np.float32))

        label = "tutorial-render" if render_pass else "tutorial-read"
        prog = Progress(label, total=total)
        i, reads = 0, 0
        while True:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                ids, mask_logits = predictor.track(frame)
            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)
            if masks.ndim == 2:
                masks = masks[None, ...]
            masks = np.array([
                sv.filter_segments_by_distance(m, relative_distance=0.03, mode="edge")
                for m in masks])
            players = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks,
                                    tracker_id=np.asarray(ids, dtype=np.int32))

            if not render_pass and i % NUMBER_EVERY == 0:
                h, w = frame.shape[:2]
                res = detector.infer(frame, confidence=PLAYER_CONFIDENCE,
                                     iou_threshold=PLAYER_IOU_THRESHOLD)[0]
                nums = sv.Detections.from_inference(res)
                nums = nums[nums.class_id == NUMBER_CLASS_ID]
                if len(nums) and len(players):
                    nums.mask = sv.xyxy_to_mask(boxes=nums.xyxy, resolution_wh=(w, h))
                    ios = sv.mask_iou_batch(masks_true=players.mask,
                                            masks_detection=nums.mask,
                                            overlap_metric=sv.OverlapMetric.IOS)
                    pairs = [(int(a), int(b)) for a, b in zip(*np.where(ios > IOS_THRESHOLD))]
                    if pairs:
                        p_idx, n_idx = zip(*pairs)
                        crops_n = [sv.crop_image(frame, xyxy) for xyxy in sv.clip_boxes(
                            sv.pad_boxes(xyxy=nums.xyxy, px=10, py=10), (w, h))]
                        values = [ocr.predict(crops_n[j], OCR_PROMPT)[0] for j in n_idx]
                        reads += len(values)
                        number_validator.update(
                            tracker_ids=[int(players.tracker_id[j]) for j in p_idx],
                            values=values)

            if render_pass and sink is not None:
                keep = players[players.area > 100]
                teams = np.array(team_validator.get_validated(
                    tracker_ids=keep.tracker_id.tolist()))
                teams = np.array([0 if t is None else int(t) for t in teams])
                numbers = number_validator.get_validated(
                    tracker_ids=keep.tracker_id.tolist())
                labels = []
                for number, team in zip(numbers, teams):
                    if number is None:
                        labels.append("")
                        continue
                    club = club_by_cluster[int(team)]
                    labels.append(f"#{number} {names_by_club[club].get(str(number), '')}".strip())
                out_frame = frame.copy()
                out_frame = annotators["mask"].annotate(
                    scene=out_frame, detections=keep, custom_color_lookup=teams)
                out_frame = annotators["label"].annotate(
                    scene=out_frame, detections=keep, labels=labels,
                    custom_color_lookup=teams)
                sink.write_frame(out_frame)

            i += 1
            prog.step(note=f"frame {i}" + ("" if render_pass else f", {reads} reads"))
            ok, frame = cap.read()
            if not ok:
                break
        cap.release()
        prog.done(note=f"{i} frames")
        return i, reads

    frames_seen, reads = prompt_and_track(render_pass=False)
    confirmed = {tid: number_validator.get_validated(tid)
                 for tid in sorted(number_validator._validated)}
    confirmed = {k: v for k, v in confirmed.items() if v is not None}
    print(f"\n{reads} OCR reads -> {len(confirmed)} tracks confirmed by "
          f"3 consecutive identical readings")
    for tid, number in confirmed.items():
        team = team_validator.get_validated(tid)
        club = club_by_cluster.get(int(team)) if team is not None else None
        name = names_by_club.get(club, {}).get(str(number), "")
        print(f"  track {tid:>3}: #{number:<4} {club or '?':<9} {name}")

    out_json = path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "video": args.video,
        "policy": "notebook: mask IoS>=0.9, 3 consecutive identical readings",
        "cluster_to_club": club_by_cluster,
        "identities": {
            str(tid): {
                "number": str(number),
                "club": club_by_cluster.get(int(team_validator.get_validated(tid) or 0)),
                "name": names_by_club.get(
                    club_by_cluster.get(int(team_validator.get_validated(tid) or 0)), {}
                ).get(str(number)),
            } for tid, number in confirmed.items()},
    }), encoding="utf-8")
    print(f"wrote {out_json}")

    # ---- cell 72: render ---------------------------------------------------
    team_colours = sv.ColorPalette.from_hex(["#C8102E", "#F58426"])
    annotators = {
        "mask": sv.MaskAnnotator(color=team_colours, opacity=0.5,
                                 color_lookup=sv.ColorLookup.INDEX),
        "label": sv.LabelAnnotator(color=team_colours, text_color=sv.Color.WHITE,
                                   text_position=sv.Position.BOTTOM_CENTER,
                                   color_lookup=sv.ColorLookup.INDEX),
    }
    info = sv.VideoInfo.from_video_path(video)
    out_video = path(args.out)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(out_video), info) as sink:
        prompt_and_track(render_pass=True, sink=sink, annotators=annotators)
    print(f"wrote {out_video}")


if __name__ == "__main__":
    main()
