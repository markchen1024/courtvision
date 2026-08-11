"""Count the lineup on the prompt frame before an hour of GPU is spent on it.

SAM2 takes its prompts once, on frame 0, and never adds another. So the set of
players that run can ever identify is decided by one detection call -- and the
whole rest of the pipeline, an hour of it, cannot recover anyone missing from
that frame.

Measured on this footage, three separate attempts were doomed at that call and
found out at the final render:

    30:06     9 prompts   ~56 min wasted
    30:16     9 prompts   ~56 min wasted
    30:05.5   9 prompts   ~56 min wasted

The number was printed every time. Nobody read it. This makes it a gate:

    python pipeline/check_lineup.py --video web/media/det_final.mp4
    python pipeline/check_lineup.py --video web/media/det.mp4 --scan --need 10

Exit status is 0 when the frame carries a full lineup and 1 when it does not,
so it can guard a run:

    python pipeline/check_lineup.py --video X && python pipeline/track_sam2_tutorial.py --video X

--scan reads the whole clip and reports which moments do carry a full lineup,
which is how a segment start should be chosen -- not by where the cut is.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import config
from progress import Progress

DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
CONFIDENCE = 0.4
IOU_THRESHOLD = 0.9
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]
ON_COURT = 10                      # five a side


def dedupe(boxes, scores, iou_thresh=0.7):
    """The same greedy NMS track_sam2.py applies to its prompts, so the count
    reported here is the count that run will actually get.

    0.7, not 0.5: players guard each other body to body, so two boxes on two
    different people overlap a lot. At 0.5 this gate reported the tutorial's
    own 04.44-04.39 sample as short a player when all eleven detections were
    real -- the shortage was ours, not the footage's.
    """
    order = np.argsort(-np.asarray(scores))
    kept = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        clash = False
        for j in kept:
            a1, b1, a2, b2 = boxes[j]
            ix = max(0.0, min(x2, a2) - max(x1, a1))
            iy = max(0.0, min(y2, b2) - max(y1, b1))
            inter = ix * iy
            union = (x2-x1)*(y2-y1) + (a2-a1)*(b2-b1) - inter
            if union and inter / union > iou_thresh:
                clash = True
                break
        if not clash:
            kept.append(int(i))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--at", type=int, default=0, help="frame to check")
    ap.add_argument("--need", type=int, default=ON_COURT,
                    help="players required for the gate to pass")
    ap.add_argument("--render", metavar="JPG",
                    help="draw the frame with every detection: green becomes a "
                         "prompt, red was struck out as a duplicate. A count is "
                         "not enough to tell a missing player from a false "
                         "positive from a de-duplication mistake, and those "
                         "three want different answers.")
    ap.add_argument("--scan", action="store_true",
                    help="read the whole clip and list frames with a full lineup")
    ap.add_argument("--every", type=int, default=30, help="scan stride")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    config.load_env()
    config.inference_env()

    import cv2
    import supervision as sv
    from inference import get_model

    video = Path(args.video)
    if not video.is_absolute():
        video = root / video
    model = get_model(model_id=DETECTION_MODEL_ID)

    def detect(frame):
        res = model.infer(frame, confidence=CONFIDENCE,
                          iou_threshold=IOU_THRESHOLD)[0]
        det = sv.Detections.from_inference(res)
        det = det[np.isin(det.class_id, PLAYER_CLASS_IDS)]
        if len(det) == 0:
            return det, set()
        return det, set(dedupe(det.xyxy.tolist(), det.confidence.tolist()))

    def count(frame):
        det, keep = detect(frame)
        return len(det), len(keep)

    def render(frame, det, keep, verdict, out_path):
        for i, (box, conf) in enumerate(zip(det.xyxy, det.confidence)):
            x1, y1, x2, y2 = (int(v) for v in box)
            kept = i in keep
            colour = (0, 210, 0) if kept else (0, 0, 240)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(frame, f"{conf:.2f}" + ("" if kept else " DUP"),
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colour, 2)
        cv2.putText(frame, f"{len(det)} detections, {len(keep)} prompts, "
                           f"{args.need} needed -- {verdict}",
                    (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"rendered {out_path}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.scan:
        rows = []
        idx = 0
        prog = Progress("check-lineup", total=total // args.every)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.every == 0:
                raw, kept = count(frame)
                rows.append((idx, raw, kept))
                prog.step(note=f"frame {idx}: {kept}")
            idx += 1
        cap.release()
        prog.done(note=f"{len(rows)} sampled")

        full = [r for r in rows if r[2] >= args.need]
        print(f"\n{len(full)} of {len(rows)} sampled frames carry {args.need}+ "
              f"players after de-duplication\n")
        if full:
            print(f"{'frame':>7} {'time':>9} {'raw':>5} {'prompts':>8}")
            for idx, raw, kept in full[:20]:
                print(f"{idx:>7} {idx/fps:>8.2f}s {raw:>5} {kept:>8}")
            best = max(rows, key=lambda r: r[2])
            print(f"\nbest: frame {best[0]} at {best[0]/fps:.2f}s "
                  f"with {best[2]} prompts")
            print(f"  ffmpeg -ss {best[0]/fps:.2f} ...   "
                  f"(then re-check the cut-free run from there)")
        else:
            best = max(rows, key=lambda r: r[2])
            print(f"no frame reaches {args.need}. Best is frame {best[0]} "
                  f"({best[0]/fps:.2f}s) with {best[2]}.")
            print("A segment starting anywhere here cannot identify a full "
                  "lineup, whatever the rest of the pipeline does.")
        return 0 if full else 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.at)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {args.at}")
    det, keep = detect(frame)
    raw, kept = len(det), len(keep)

    print(f"frame {args.at} of {video.name}")
    print(f"  {raw} player detections at confidence {CONFIDENCE}")
    print(f"  {kept} prompts after de-duplication")
    print(f"  {args.need} needed\n")
    if args.render:
        render(frame.copy(), det, keep,
               "PASS" if kept >= args.need else "FAIL", args.render)
    if kept >= args.need:
        print(f"PASS -- a run from here can reach all {args.need} players.")
        return 0
    print(f"FAIL -- short by {args.need - kept}.")
    print("SAM2 only takes prompts on this frame, so those players cannot be")
    print("identified later by any stage downstream. Roughly an hour of GPU")
    print("would produce a result that is missing them.")
    print("\n  pipeline/check_lineup.py --video ... --scan    # find a frame that passes")
    return 1


if __name__ == "__main__":
    sys.exit(main())
