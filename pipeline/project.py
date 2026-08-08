"""Turn detections into court coordinates and write what the viewer reads.

This is where the geometry finally pays off. Each frame already has a homography
back to the calibrated hub, so composing gives court metres per frame:

    H_frame_to_court = inverse( H_ref_to_frame @ H_court_to_ref )

A player's contact with the floor is the bottom-centre of their box, and that is
the only point of a standing person that lies on the court plane -- project the
centre of the box instead and you are projecting a point floating a metre in the
air, which the homography will happily place several metres up the floor.

Two things fall out of working in court metres rather than pixels:

  the bench filters itself   anyone whose feet land outside the lines is not on
                             the court, so substitutes, coaches and the front row
                             drop out without a single heuristic about where in
                             the frame they happen to be.
  tracking gets easier       the camera pans, so image-space association fights
                             the camera the whole way. Court space has the camera
                             motion already divided out, and players move
                             smoothly and slowly across it.

    python pipeline/project.py --calibration out/calibration/fused_clean.json

No jersey numbers: that is OCR, which is the part Superstat has spent two years
on, so tracks are labelled T1, T2 ... and nothing here pretends otherwise. No
events either -- those get tagged by hand.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate import COURT_LENGTH, COURT_WIDTH

DETECTOR = "koppolusameer/rfdetr-basketball-player-ball-referee-detection"

# Detection is the slow half and the tuning knobs all live downstream of it, so
# cache the raw boxes once at a permissive threshold and filter at assembly time.
# Otherwise every tracker experiment costs another ten minutes of GPU.
CACHE_THRESHOLD = 0.25

# Fashion-CLIP, used for its image embedding rather than its zero-shot head: the
# tutorials prompt it with colour names, which means knowing the strips in
# advance, and clustering embeddings does not.
#
# It is not the default, because it measured no better. On the frames where all
# ten players are detected -- the only ones where the split can be judged --
# embeddings and mean chroma both leave a median imbalance of 2 and get a clean
# 5v5 in 24% of frames, identically. The shirt descriptor is not the bottleneck:
# only 90 of 600 frames detect exactly ten players, and 128 detect eleven to
# thirteen, which is the on-court margin letting the bench in. Both numbers trace
# back to calibration precision.
CLIP_MODEL = "patrickjohncyh/fashion-clip"

# A foot position is only believable if it lands on the floor we calibrated, plus
# slack. 1.5m is not a physical allowance -- it is the measured size of the
# calibration error: tighten it to 0.5m and 516 detections fall outside the lines
# instead of 254, and the per-frame count drops from 9 to 7. Players who are
# plainly on the court are landing up to a metre and a half off it.
MARGIN = 1.5

# Association gate. A player covers ~1.3m between samples at 6Hz flat out, so the
# physical gate is about 1.5m; 3.5 buys room for the position jitter the
# calibration error causes. Beyond that the gate stops meaning anything -- 5m here
# would be 30 m/s, and it "helps" only by joining tracks that are not the same
# player.
GATE_M = 3.5
MAX_MISSES = 30


def load_detector():
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    import torch

    proc = AutoImageProcessor.from_pretrained(DETECTOR)
    model = AutoModelForObjectDetection.from_pretrained(DETECTOR)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return proc, model


def detect(proc, model, frame, threshold):
    import torch

    inputs = proc(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor([[frame.shape[0], frame.shape[1]]])
    )[0]
    dets = []
    for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
        name = model.config.id2label.get(int(label), str(int(label))).lower()
        dets.append((name, float(score), [float(v) for v in box.tolist()]))
    return dets


def jersey_colour(frame, box):
    """Mean Lab colour of the torso -- the top third of the box, middle half."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    cx0 = int(max(0, x1 + (x2 - x1) * 0.25))
    cx1 = int(min(w, x1 + (x2 - x1) * 0.75))
    cy0 = int(max(0, y1 + (y2 - y1) * 0.15))
    cy1 = int(min(h, y1 + (y2 - y1) * 0.45))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    patch = cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2LAB)
    return patch.reshape(-1, 3).mean(axis=0)


def clip_embeddings(cache, video, path, batch=64):
    """One 512-d vector per player box. Cached: it is a second pass over the video."""
    import numpy as _np
    if Path(path).exists():
        z = _np.load(path)
        return {tuple(k): v for k, v in zip(z["keys"], z["vecs"])}

    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    if torch.cuda.is_available():
        model = model.cuda()

    cap = cv2.VideoCapture(video)
    keys, crops, vecs = [], [], []

    def flush():
        if not crops:
            return
        inputs = proc(images=crops, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            f = model.get_image_features(**inputs)
        # transformers 5 hands back an output object here, not a tensor; the
        # projected embedding is pooler_output.
        if not torch.is_tensor(f):
            f = f.pooler_output
        f = torch.nn.functional.normalize(f, dim=-1)
        vecs.extend(f.cpu().numpy())
        crops.clear()

    for n, (frame_idx, rows) in enumerate(sorted(cache.items())):
        players = [(i, r) for i, r in enumerate(rows) if r["name"] == "player"]
        if not players:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        for i, r in players:
            x1, y1, x2, y2 = r["box"]
            # torso only: legs and floor dilute the shirt
            cx0, cx1 = int(max(0, x1)), int(min(w, x2))
            cy0 = int(max(0, y1 + (y2 - y1) * 0.10))
            cy1 = int(min(h, y1 + (y2 - y1) * 0.55))
            if cx1 - cx0 < 4 or cy1 - cy0 < 4:
                continue
            keys.append((frame_idx, i))
            crops.append(cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB))
            if len(crops) >= batch:
                flush()
        if (n + 1) % 100 == 0:
            print(f"  embedding {n + 1}/{len(cache)} frames...")
    flush()
    cap.release()

    arr = _np.array(vecs, _np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _np.savez_compressed(path, keys=_np.array(keys, _np.int64), vecs=arr)
    print(f"cached {len(arr)} CLIP embeddings to {path}")
    return {k: v for k, v in zip(keys, arr)}


def two_means(x, iters=40, seed=0):
    """Two clusters, no sklearn. Seeded from the two most distant samples so the
    result does not depend on a random draw across runs."""
    x = np.asarray(x, np.float64)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    c = x[[i, j]].copy()
    lab = np.zeros(len(x), int)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(x[:, None, :] - c[None, :, :], axis=-1), axis=1)
        for k in (0, 1):
            if (lab == k).any():
                c[k] = x[lab == k].mean(axis=0)
    return lab, c


class CourtTracker:
    """Greedy nearest-neighbour association in metres, with a constant-velocity
    prediction. Crude, but the hard part of tracking -- the camera moving under
    the targets -- has already been removed by the homography."""

    def __init__(self, gate=GATE_M, max_misses=MAX_MISSES):
        self.tracks = {}
        self.next_id = 1
        self.gate = gate
        self.max_misses = max_misses
        # Shirt colours live here, not on the track: tracks get deleted when they
        # go stale, and taking the palette off the survivors silently defaults
        # every finished track to one team.
        self.palette = {}

    def step(self, points, colours):
        preds = {tid: (t["x"] + t["vx"], t["y"] + t["vy"]) for tid, t in self.tracks.items()}
        pairs = []
        for pi, (px, py) in enumerate(points):
            for tid, (qx, qy) in preds.items():
                dist = float(np.hypot(px - qx, py - qy))
                if dist <= self.gate:
                    pairs.append((dist, pi, tid))
        pairs.sort()

        taken_p, taken_t, assign = set(), set(), {}
        for dist, pi, tid in pairs:
            if pi in taken_p or tid in taken_t:
                continue
            taken_p.add(pi)
            taken_t.add(tid)
            assign[pi] = tid

        for tid in self.tracks:
            self.tracks[tid]["missed"] = 0 if tid in taken_t else self.tracks[tid]["missed"] + 1

        out = []
        for pi, (px, py) in enumerate(points):
            tid = assign.get(pi)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"x": px, "y": py, "vx": 0.0, "vy": 0.0, "missed": 0}
            t = self.tracks[tid]
            t["vx"], t["vy"] = 0.6 * t["vx"] + 0.4 * (px - t["x"]), 0.6 * t["vy"] + 0.4 * (py - t["y"])
            t["x"], t["y"] = px, py
            if colours[pi] is not None:
                self.palette.setdefault(tid, []).append(colours[pi])
            out.append((tid, px, py))

        for tid in [k for k, v in self.tracks.items() if v["missed"] > self.max_misses]:
            del self.tracks[tid]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--registration", default="out/registration/registration.json")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--cache", default="out/detections.json")
    ap.add_argument("--redetect", action="store_true", help="ignore the detection cache")
    ap.add_argument("--max-misses", type=int, default=MAX_MISSES,
                    help="samples a track survives unmatched (6Hz, so 30 is 5s)")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="metres outside the lines still counted as on court")
    ap.add_argument("--gate", type=float, default=GATE_M, help="association gate, metres")
    ap.add_argument("--team", choices=["colour", "clip"], default="colour",
                    help="shirt description used to split the two sides; clip is "
                         "available but measured no better here")
    ap.add_argument("--embeddings", default="out/embeddings.npz")
    ap.add_argument("--min-samples", type=int, default=8,
                    help="drop tracks seen fewer times than this")
    ap.add_argument("--out", default="web/data/sample.json")
    args = ap.parse_args()

    cal = json.loads(Path(args.calibration).read_text())
    if cal.get("H_court_to_image") is None:
        raise SystemExit(f"{args.calibration} is a partial calibration with no homography")
    H_court_to_ref = np.array(cal["H_court_to_image"], np.float64)
    print(f"calibration: frame {cal['frame']}, {len(cal['points'])} points, "
          f"rms {cal['reprojection_rms_px']:.2f}px, coverage {cal['coverage_m2']:.0f}m2")

    reg = json.loads(Path(args.registration).read_text())
    if reg["reference_frame"] != cal["frame"]:
        raise SystemExit(f"registration hubs on frame {reg['reference_frame']} but the "
                         f"calibration is for frame {cal['frame']}")
    usable = [r for r in reg["frames"] if "H_ref_to_frame" in r]
    fps = 30.0
    hz = fps / reg["every"]
    print(f"registration: {len(usable)} frames at {hz:.1f}Hz, hub frame {reg['reference_frame']}")

    cache_path = Path(args.cache)
    cache = None
    if cache_path.exists() and not args.redetect:
        c = json.loads(cache_path.read_text())
        if (c.get("video") == args.video and c.get("frames_hash") == len(usable)
                and c.get("threshold") <= CACHE_THRESHOLD + 1e-9):
            cache = {int(k): v for k, v in c["detections"].items()}
            print(f"detections: reusing {cache_path} ({len(cache)} frames at >={c['threshold']})")

    if cache is None:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {args.video}")
        print(f"loading {DETECTOR}")
        proc, model = load_detector()
        cache = {}
        for n, r in enumerate(usable):
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            rows = []
            for name, score, box in detect(proc, model, frame, CACHE_THRESHOLD):
                colour = jersey_colour(frame, box) if name == "player" else None
                rows.append({"name": name, "score": score, "box": box,
                             "colour": None if colour is None else colour.tolist()})
            cache[r["frame"]] = rows
            if (n + 1) % 50 == 0:
                print(f"  detecting {n + 1}/{len(usable)} frames...")
        cap.release()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "video": args.video, "threshold": CACHE_THRESHOLD, "frames_hash": len(usable),
            "detections": {str(k): v for k, v in cache.items()},
        }))
        print(f"cached raw detections to {cache_path}")

    embeddings = {}
    if args.team == "clip":
        embeddings = clip_embeddings(cache, args.video, args.embeddings)

    tracker = CourtTracker(gate=args.gate, max_misses=args.max_misses)
    frames_out, kept, dropped_off_court, dropped_class, dropped_score = [], 0, 0, 0, 0

    for r in usable:
        H_court_to_frame = np.array(r["H_ref_to_frame"], np.float64) @ H_court_to_ref
        try:
            H_frame_to_court = np.linalg.inv(H_court_to_frame)
        except np.linalg.LinAlgError:
            continue

        pts, cols = [], []
        for di, det in enumerate(cache.get(r["frame"], [])):
            if det["name"] != "player":
                dropped_class += 1
                continue
            if det["score"] < args.threshold:
                dropped_score += 1
                continue
            x1, y1, x2, y2 = det["box"]
            foot = np.array([[[(x1 + x2) / 2.0, y2]]], np.float32)
            cx, cy = cv2.perspectiveTransform(foot, H_frame_to_court).reshape(2)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            if not (-args.margin <= cx <= COURT_LENGTH + args.margin
                    and -args.margin <= cy <= COURT_WIDTH + args.margin):
                dropped_off_court += 1
                continue
            pts.append((float(cx), float(cy)))
            if args.team == "clip":
                cols.append(embeddings.get((r["frame"], di)))
            else:
                cols.append(None if det["colour"] is None else np.array(det["colour"]))
            kept += 1

        assigned = tracker.step(pts, cols)
        frames_out.append({
            "t": round(r["frame"] / fps, 2),
            "positions": [{"id": tid, "x": round(x, 2), "y": round(y, 2)} for tid, x, y in assigned],
        })

    counts = {}
    for f in frames_out:
        for p in f["positions"]:
            counts[p["id"]] = counts.get(p["id"], 0) + 1
    live = {tid for tid, c in counts.items() if c >= args.min_samples}
    for f in frames_out:
        f["positions"] = [p for p in f["positions"] if p["id"] in live]

    # Team from shirt colour. Two teams, so two clusters; which is "home" is
    # arbitrary and the viewer only uses it to pick a colour.
    # colour: chroma only, dropping L. Lightness carries 90% of the variance in
    # these torso patches and almost all of it is illumination and whether we are
    # seeing the front or the back of a shirt, so clustering on it files a
    # shadowed white jersey with the blue team.
    # clip: the whole embedding, re-normalised after averaging over the track.
    def describe(vs):
        m = np.mean(vs, axis=0)
        if args.team == "clip":
            n = np.linalg.norm(m)
            return m / n if n > 0 else m
        return m[1:]

    palette = {tid: describe(tracker.palette[tid])
               for tid in live if tracker.palette.get(tid)}
    teams = {}
    if len(palette) >= 2:
        ids = sorted(palette)
        lab, _ = two_means([palette[i] for i in ids])
        teams = {i: ("home" if k == 0 else "away") for i, k in zip(ids, lab)}

    players = [{"id": tid, "team": teams.get(tid, "home"), "number": f"T{tid}"}
               for tid in sorted(live)]

    doc = {
        "source": "detected positions, hand-tagged events pending",
        "video": {"duration": round(usable[-1]["frame"] / fps, 2), "hz": hz},
        "court": {"length_m": COURT_LENGTH, "width_m": COURT_WIDTH},
        "players": players,
        "frames": frames_out,
        "events": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")

    per_frame = np.array([len(f["positions"]) for f in frames_out])
    print(f"\nwrote {out}")
    print(f"  {len(frames_out)} frames, {len(players)} tracks surviving the {args.min_samples}-sample floor")
    print(f"  players on court per frame: median {np.median(per_frame):.1f}  "
          f"min {per_frame.min()}  max {per_frame.max()}   (10 is right)")
    print(f"  dropped {dropped_class} non-player detections (referee, ball)")
    print(f"  dropped {dropped_score} player boxes below the {args.threshold} score threshold")
    print(f"  dropped {dropped_off_court} detections whose feet landed off the court")
    lifetimes = np.array([counts[t] for t in live])
    print(f"  identity holds {np.median(lifetimes) / hz:.1f}s before a track breaks or switches")
    print(f"\nthe {args.margin:.1f}m margin is absorbing calibration error, not sideline scramble:")
    print("tighten it to 0.5m and the per-frame count falls from 9 to 7. That is the")
    print("measurement -- players plainly on the court are landing off it.")
    print("Closing the identity gap needs appearance re-identification, a different problem")
    print("from geometry, and it is not attempted here.")


if __name__ == "__main__":
    main()
