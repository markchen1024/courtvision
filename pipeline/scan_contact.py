"""Rank windows by player contact. VALIDATED AND FOUND WANTING -- see below.

DO NOT use this to pick segments. It was written to predict collapses cheaply
and it does not. Kept because the way it fails is worth remembering.

Checked against the three segments that have ground truth:

    seg_02m28.00s_13s   0 sustained contacts   correct, it scores 100/100
    seg_02m44.15s_10s   0 sustained contacts   correct, it scores 100/99.8
    seg_01m10.87s_19s   1 contact at 4.7-5.5s  WRONG EVENT

The 19s segment's collapse is at 11.9s and this finds nothing there. At 11.68s
the detector returns no pair above IoS 0.75, while SAM2's masks for tracks 6
and 8 are already at 0.98 and stay there for seven seconds. The detector sees
Brunson and Beasley as two separate men; the tracker has merged them. What it
did find, at 4.7-5.5s, is the Towns/Harris post-up that count_players had
already classified as ordinary occlusion.

So a collapse is not predicted by contact visible to the detector at the time
it happens. Two right answers out of three looked like a working proxy, and it
would have shortlisted windows that collapse.

What does work is measuring the thing itself: track_sam2_tutorial.py then
overlap.py --video, about seven minutes a segment against twenty-five for the
full pipeline, because overlap.py compares tracker boxes and asks the detector
how many men are inside them -- which is the quantity that produces the fault.

Everything below still runs and the contact numbers are real. They are simply
not the numbers that decide whether a segment is usable.

---

Rank windows of footage by player contact.

Five segments have now been run end to end and scored against ground truth,
and the dividing line is not the detector, the OCR or the roster. It is
whether two players spend a second or more pressed together:

    seg_02m28.00s_13s   no sustained contact          100% / 100%
    seg_02m44.15s_10s   no sustained contact          100% /  99.8%
    seg_01m10.87s_19s   one 7.3s contact at 11.9s     100% /  91.7%
    seg_01m48.88s_33s   one 15s contact at 18.2s       -   /  90%
    seg_00m03.54s_26s   two contacts                   -   /  74%

When two men merge, SAM2 has no way to tell which one it kept, and no gate
downstream can undo that -- the most it can do is refuse to name either. So
the cheapest useful thing is to find the windows where it never happens.

That question needs the detector and nothing else. No SAM2, no OCR, no
identify: sample a few frames a second, keep the boxes standing on the floor,
and look for pairs that overlap at IoS >= 0.75 for a second or more. A
twenty-second window scores in a minute or two instead of the twenty-five that
the full pipeline costs, so the whole clip can be swept before committing GPU
to any of it.

Contacts are followed without track ids, by the position of the pair: a
contact in this sample continues the one in the last if their union boxes sit
within --link-px of each other.

    python pipeline/scan_contact.py --video web/media/det.mp4 --window 12
    python pipeline/scan_contact.py --video X --from 40 --to 110 --window 15

Prints windows best-first with the ffmpeg command to cut each one.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import oncourt
import overlap
from check_lineup import CONFIDENCE, IOU_THRESHOLD, PLAYER_CLASS_IDS, dedupe

ON_COURT = 10
CONTACT_IOS = overlap.IOS_THRESHOLD      # the same 0.75 the collapse test uses
MIN_CONTACT_S = 1.0                      # and the same one-second floor


def centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def sample(video, every, court_margin, start=0, end=None, progress=None):
    """Per sampled frame: how many are on the floor, and which pairs touch."""
    import cv2
    import supervision as sv
    from inference import get_model

    det = get_model(model_id="basketball-player-detection-3-ycjdo/4")
    kp = oncourt.keypoint_model()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end or total, total)

    out = {}
    transformer = None
    f = start
    while f < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        res = det.infer(frame, confidence=CONFIDENCE,
                        iou_threshold=IOU_THRESHOLD)[0]
        d = sv.Detections.from_inference(res)
        d = d[np.isin(d.class_id, PLAYER_CLASS_IDS)]
        if len(d):
            on, _, _ = oncourt.feet_on_court(frame, d.xyxy,
                                             margin_m=court_margin, model=kp)
            d = d[on]
        if len(d):
            d = d[dedupe(d.xyxy.tolist(), d.confidence.tolist())]
        boxes = [[float(v) for v in b] for b in d.xyxy]
        pairs = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if overlap.ios(boxes[i], boxes[j]) >= CONTACT_IOS:
                    pairs.append(union(boxes[i], boxes[j]))
        out[f] = {"players": len(boxes), "contacts": pairs}
        if progress:
            progress.step(note=f"frame {f}, {len(boxes)} players, "
                               f"{len(pairs)} contacts")
        f += every
    cap.release()
    return out, fps


def contact_runs(samples, every, fps, link_px):
    """Chain per-frame contacts into events, without any track ids."""
    frames = sorted(samples)
    live, done = [], []
    for f in frames:
        fresh = samples[f]["contacts"]
        used = set()
        for ev in live:
            best, bestd = None, link_px
            for k, box in enumerate(fresh):
                if k in used:
                    continue
                dx, dy = np.subtract(centre(box), centre(ev["box"]))
                dist = float(np.hypot(dx, dy))
                if dist < bestd:
                    best, bestd = k, dist
            if best is None:
                ev["dead"] = True
            else:
                used.add(best)
                ev["box"] = fresh[best]
                ev["last"] = f
        done += [e for e in live if e.get("dead")]
        live = [e for e in live if not e.get("dead")]
        for k, box in enumerate(fresh):
            if k not in used:
                live.append({"box": box, "first": f, "last": f})
    done += live
    return [{"from": e["first"] / fps, "to": e["last"] / fps,
             "seconds": (e["last"] - e["first"] + every) / fps}
            for e in done if (e["last"] - e["first"] + every) / fps >= MIN_CONTACT_S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--window", type=float, default=12.0,
                    help="seconds per candidate window")
    ap.add_argument("--stride", type=float, default=0.0,
                    help="seconds between window starts; default is half a window")
    ap.add_argument("--every", type=int, default=10,
                    help="frames between samples; 10 at 60fps is 6Hz, enough to "
                         "see a one-second contact")
    ap.add_argument("--from", dest="start_s", type=float, default=0.0)
    ap.add_argument("--to", dest="end_s", type=float, default=0.0)
    ap.add_argument("--link-px", type=float, default=150.0,
                    help="how far a contact may move between samples and still "
                         "be the same event")
    ap.add_argument("--court-margin", type=float, default=0.0,
                    help="metres past the lines a detection may stand. 0 keeps "
                         "the bench out, which is what per-frame filtering "
                         "wants -- see oncourt.py")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="out/contact_scan.json")
    args = ap.parse_args()

    config.load_env()
    config.inference_env()
    from progress import Progress

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video

    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start = int(args.start_s * fps)
    end = int(args.end_s * fps) if args.end_s else total

    n = (end - start) // args.every
    prog = Progress("scan-contact", total=n, video=args.video)
    samples, fps = sample(video, args.every, args.court_margin, start, end, prog)
    prog.done(note=f"{len(samples)} frames sampled")

    events = contact_runs(samples, args.every, fps, args.link_px)
    print(f"\nsampled {len(samples)} frames over "
          f"{(end - start) / fps:.0f}s; {len(events)} sustained contacts "
          f"(IoS >= {CONTACT_IOS} for {MIN_CONTACT_S}s+)")
    for e in sorted(events, key=lambda e: -e["seconds"])[:10]:
        print(f"    {e['from']:6.1f}s - {e['to']:6.1f}s   {e['seconds']:4.1f}s")

    stride = args.stride or args.window / 2
    windows = []
    t = start / fps
    while t + args.window <= end / fps:
        a, b = t, t + args.window
        inside = [e for e in events if e["to"] > a and e["from"] < b]
        contact_s = sum(min(e["to"], b) - max(e["from"], a) for e in inside)
        fs = [v for f, v in samples.items() if a <= f / fps < b]
        short = sum(1 for v in fs if v["players"] < ON_COURT)
        windows.append({
            "from": round(a, 2), "to": round(b, 2),
            "contact_seconds": round(contact_s, 2),
            "contacts": len(inside),
            "short_share": round(short / max(1, len(fs)), 3),
            "min_players": min((v["players"] for v in fs), default=0),
            "mean_players": round(sum(v["players"] for v in fs) / max(1, len(fs)), 2),
        })
        t += stride

    windows.sort(key=lambda w: (w["contact_seconds"], w["short_share"],
                                -w["mean_players"]))
    Path(ROOT / args.out).write_text(json.dumps(
        {"video": args.video, "window": args.window, "events": events,
         "windows": windows}, indent=1))

    print(f"\nbest {min(args.top, len(windows))} windows of {args.window:.0f}s "
          f"(least contact first):\n")
    print(f"{'from':>8} {'to':>8} {'contact':>8} {'short':>7} {'min':>4} {'mean':>5}")
    for w in windows[:args.top]:
        print(f"{w['from']:>7.1f}s {w['to']:>7.1f}s {w['contact_seconds']:>7.1f}s "
              f"{w['short_share']:>6.0%} {w['min_players']:>4} {w['mean_players']:>5.1f}")
    if windows:
        w = windows[0]
        print(f"\n  ffmpeg -y -ss {w['from']:.2f} -i {args.video} "
              f"-t {args.window:.2f} -c:v libx264 -crf 18 -preset veryfast "
              f"out/segments/seg_scan_{w['from']:.0f}s.mp4")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
