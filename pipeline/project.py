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

    python pipeline/project.py --auto-calibration out/auto_calibration.json         --video web/media/nba.mp4

Tracks leave here labelled T1, T2 ...; identify.py then names them with jersey
OCR where it can. Events are tagged by hand. The tracker below is our own code,
not the tutorial's SAM2 -- see CourtTracker's docstring for that deviation.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate import COURT_LENGTH, COURT_WIDTH
from progress import Progress

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


# The colours a basketball strip is actually likely to be. The pair that splits
# the crops most cleanly wins, so nothing has to be configured per fixture.
SHIRT_WORDS = ["white", "black", "blue", "red", "green", "yellow", "grey", "navy",
               "orange", "purple", "maroon", "cream"]


def clip_zero_shot(cache, video, path, batch=64):
    """Score every player crop against a colour vocabulary, the way the two
    published basketball pipelines do it.

    This is the part that was reinvented and should not have been. Fashion-CLIP
    was already wired in, but only for its embedding, which then went into a
    k-means of my own -- and that k-means is what collapsed to a 119/21 split on
    Summer League. The model's own zero-shot head is the method the tutorials
    use, and the separation measurement says the signal is there to be had:
    between/within ratio 3.1 on this clip, 2.9 on the community one.
    """
    if Path(path).exists():
        z = np.load(path, allow_pickle=True)
        if list(z["words"]) == SHIRT_WORDS:
            return {tuple(k): v for k, v in zip(z["keys"], z["scores"])}

    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    if torch.cuda.is_available():
        model = model.cuda()

    prompts = [f"a basketball player wearing a {w} jersey" for w in SHIRT_WORDS]
    tin = proc(text=prompts, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        tin = {k: v.cuda() for k, v in tin.items()}
    with torch.no_grad():
        tf = model.get_text_features(**tin)
        if not torch.is_tensor(tf):
            tf = tf.pooler_output
    tf = torch.nn.functional.normalize(tf, dim=-1)

    cap = cv2.VideoCapture(video)
    keys, crops, scores = [], [], []

    def flush():
        if not crops:
            return
        inputs = proc(images=crops, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            f = model.get_image_features(**inputs)
        if not torch.is_tensor(f):
            f = f.pooler_output
        f = torch.nn.functional.normalize(f, dim=-1)
        scores.extend((f @ tf.T).cpu().numpy())
        crops.clear()

    prog = Progress("shirts", total=len(cache), video=video, artifact=path)
    for n, (frame_idx, rows) in enumerate(sorted(cache.items())):
        prog.step(note=f"frame {frame_idx}")
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
            print(f"  scoring shirts {n + 1}/{len(cache)} frames...")
    flush()
    cap.release()

    arr = np.array(scores, np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=np.array(keys, np.int64), scores=arr,
                        words=np.array(SHIRT_WORDS))
    print(f"cached {len(arr)} shirt scores to {path}")
    return {k: v for k, v in zip(keys, arr)}


def pick_colour_pair(track_scores):
    """Which two colour words split these tracks best.

    Every pair is scored on how far apart the two groups sit relative to their
    own spread, so an evenly-split, confident pair beats one where every track
    looks equally like both.
    """
    best, choice = -1.0, (0, 1)
    for i in range(len(SHIRT_WORDS)):
        for j in range(i + 1, len(SHIRT_WORDS)):
            d = np.array([s[i] - s[j] for s in track_scores.values()])
            side = d > 0
            if side.sum() < 2 or (~side).sum() < 2:
                continue
            spread = (d[side].std() + d[~side].std()) / 2
            sep = abs(d[side].mean() - d[~side].mean()) / max(1e-9, spread)
            balance = min(side.mean(), 1 - side.mean()) / 0.5
            if sep * balance > best:
                best, choice = sep * balance, (i, j)
    return choice, best


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
    """Two clusters, no sklearn.

    Seeded along the first principal component at the 10th and 90th percentiles,
    not at the two most distant samples. Distant-pair seeding is what the obvious
    implementation does and it fails the moment one shirt patch is an outlier --
    a referee taken for a player, someone in deep shadow -- because both seeds
    land on outliers and every real point falls into whichever is marginally
    nearer. Measured on the Summer League clip that produced a 119/21 split where
    it should be even, and no frame at all with five a side.
    """
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return np.zeros(len(x), int), x
    centred = x - x.mean(axis=0)
    # First principal component: the direction the shirts differ along.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    t = centred @ vt[0]
    lo, hi = np.percentile(t, 10), np.percentile(t, 90)
    c = np.stack([x[np.argmin(np.abs(t - lo))], x[np.argmin(np.abs(t - hi))]])

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
    the targets -- has already been removed by the homography.

    HAND-WRITTEN, and a deviation to be upfront about: the tutorial this
    pipeline follows (blog.roboflow.com/identify-basketball-players) tracks
    with SAM2, and this class is not a ready-made component. The measured
    head-to-head is in docs/tracking-comparison.md -- on broadcast footage
    with cuts, associating in court coordinates named 6 players where SAM2
    named 2. One caveat that comparison does not settle: the win comes from
    the coordinate space, not necessarily from these 50 lines. A ready-made
    tracker fed the same court coordinates (Norfair takes custom points and
    distances natively) has not been tried, and would be the honest
    replacement if it matches."""

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
    ap.add_argument("--calibration", help="a single calibrated frame, used with --registration")
    ap.add_argument("--auto-calibration", help="per-frame homographies from auto_calibrate.py")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--cache", default="out/detections.json")
    ap.add_argument("--redetect", action="store_true", help="ignore the detection cache")
    ap.add_argument("--max-misses", type=int, default=MAX_MISSES,
                    help="samples a track survives unmatched (6Hz, so 30 is 5s)")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="metres outside the lines still counted as on court")
    ap.add_argument("--gate", type=float, default=GATE_M, help="association gate, metres")
    ap.add_argument("--team", choices=["zero-shot", "colour", "clip"], default="zero-shot",
                    help="how the two sides are separated: the model's own zero-shot "
                         "head, or clustering a colour statistic")
    ap.add_argument("--embeddings", default="out/embeddings.npz")
    ap.add_argument("--tracks", help="external track ids (track_bytetrack.py output); "
                    "replaces the court-space tracker")
    ap.add_argument("--min-samples", type=int, default=8,
                    help="drop tracks seen fewer times than this")
    ap.add_argument("--boxes-out", help="also write per-frame (tid, box) rows; "
                    "identify.py and render_tracks.py consume this")
    ap.add_argument("--out", default="web/data/sample.json")
    args = ap.parse_args()

    # Two ways to know where the court is in a frame, and the pipeline downstream
    # of here does not care which. Hand-calibrate one frame and carry it with
    # feature matching, or solve every frame independently from a keypoint model.
    # The second is simpler and needs no human; it only became possible once a
    # model was accurate enough on the footage and its keypoint numbering had
    # been decoded.
    if bool(args.calibration) == bool(args.auto_calibration):
        raise SystemExit("pass exactly one of --calibration or --auto-calibration")

    if args.auto_calibration:
        auto = json.loads(Path(args.auto_calibration).read_text())
        if auto["video"] != args.video:
            raise SystemExit(f"{args.auto_calibration} was solved for {auto['video']}")
        usable = [{"frame": r["frame"], "H_court_to_image": r["H_court_to_image"]}
                  for r in auto["frames"] if r.get("solved") and not r.get("suspect")]
        fps = auto["fps"]
        hz = fps / auto["every"]
        skipped = len(auto["frames"]) - len(usable)
        print(f"auto calibration: {len(usable)} frames solved at {hz:.1f}Hz "
              f"({skipped} frames have no court in view and are left out)")
    else:
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
        usable = [{"frame": r["frame"],
                   "H_court_to_image": (np.array(r["H_ref_to_frame"], np.float64)
                                        @ H_court_to_ref).tolist()}
                  for r in reg["frames"] if "H_ref_to_frame" in r]
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
        # The detector id is recorded, not described. A docstring saying which
        # model this uses has been wrong before; a run that carries its own
        # answer cannot be.
        prog = Progress("detect", total=len(usable), video=args.video,
                        artifact=cache_path, meta={"detector": DETECTOR})
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
            prog.step(note=f"frame {r['frame']}")
            if (n + 1) % 50 == 0:
                print(f"  detecting {n + 1}/{len(usable)} frames...")
        cap.release()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prog.done()
        cache_path.write_text(json.dumps({
            "video": args.video, "threshold": CACHE_THRESHOLD, "frames_hash": len(usable),
            "detections": {str(k): v for k, v in cache.items()},
        }))
        print(f"cached raw detections to {cache_path}")

    embeddings = {}
    if args.team == "zero-shot":
        embeddings = clip_zero_shot(cache, args.video, args.embeddings)
    elif args.team == "clip":
        embeddings = clip_embeddings(cache, args.video, args.embeddings)

    # Identities come from one of two places: the court-space tracker written
    # here, or an external tracker's output (ByteTrack, the tutorial's choice).
    # The external file runs on a denser grid, so this pass adopts its ids by
    # IoU-matching each cached box against the tracked boxes at the nearest
    # tracked frame -- association happens there, only the naming happens here.
    ext_tracks, ext_every = None, 1
    if args.tracks:
        ext = json.loads(Path(args.tracks).read_text())
        if ext["video"] != args.video:
            raise SystemExit(f"{args.tracks} was tracked on {ext['video']}")
        ext_tracks = {int(k): v for k, v in ext["frames"].items()}
        ext_every = ext["every"]
        print(f"identities: {ext.get('tracker', 'external')} over {len(ext_tracks)} frames")

    def adopt_ids(frame_idx, boxes):
        near = frame_idx - (frame_idx % ext_every)
        rows = ext_tracks.get(near) or ext_tracks.get(near + ext_every) or []
        out = []
        for (x1, y1, x2, y2) in boxes:
            best_tid, best_iou = None, 0.4   # below this the boxes are not the same person
            for t in rows:
                a1, b1, a2, b2 = t["box"]
                iw = max(0.0, min(x2, a2) - max(x1, a1))
                ih = max(0.0, min(y2, b2) - max(y1, b1))
                inter = iw * ih
                union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter
                if union > 0 and inter / union > best_iou:
                    best_tid, best_iou = t["tid"], inter / union
            out.append(best_tid)
        return out

    tracker = CourtTracker(gate=args.gate, max_misses=args.max_misses)
    ext_palette = {}
    frames_out, kept, dropped_off_court, dropped_class, dropped_score = [], 0, 0, 0, 0
    dropped_unmatched = 0
    box_frames = {}

    for r in usable:
        H_court_to_frame = np.array(r["H_court_to_image"], np.float64)
        try:
            H_frame_to_court = np.linalg.inv(H_court_to_frame)
        except np.linalg.LinAlgError:
            continue

        pts, cols, boxes = [], [], []
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
            boxes.append((x1, y1, x2, y2))
            if args.team in ("clip", "zero-shot"):
                cols.append(embeddings.get((r["frame"], di)))
            else:
                cols.append(None if det["colour"] is None else np.array(det["colour"]))
            kept += 1

        if ext_tracks is not None:
            assigned, assigned_boxes = [], []
            for tid, (px, py), col, bx in zip(adopt_ids(r["frame"], boxes), pts, cols, boxes):
                if tid is None:
                    dropped_unmatched += 1   # no tracked box agrees this person exists
                    continue
                assigned.append((tid, px, py))
                assigned_boxes.append(bx)
                if col is not None:
                    ext_palette.setdefault(tid, []).append(col)
        else:
            assigned = tracker.step(pts, cols)
            assigned_boxes = boxes   # step() returns one row per point, same order
        frames_out.append({
            "t": round(r["frame"] / fps, 2),
            "positions": [{"id": tid, "x": round(x, 2), "y": round(y, 2)} for tid, x, y in assigned],
        })
        box_frames[r["frame"]] = [
            {"tid": tid, "box": [round(float(v), 1) for v in bx]}
            for (tid, _, _), bx in zip(assigned, assigned_boxes)]

    counts = {}
    for f in frames_out:
        for p in f["positions"]:
            counts[p["id"]] = counts.get(p["id"], 0) + 1
    live = {tid for tid, c in counts.items() if c >= args.min_samples}
    for f in frames_out:
        f["positions"] = [p for p in f["positions"] if p["id"] in live]

    if args.boxes_out:
        grid = sorted(box_frames)
        every = int(np.median(np.diff(grid))) if len(grid) > 1 else 1
        Path(args.boxes_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.boxes_out).write_text(json.dumps({
            "video": args.video, "fps": fps, "every": every,
            "frames": {str(k): [t for t in v if t["tid"] in live]
                       for k, v in box_frames.items()},
        }))
        print(f"wrote track boxes to {args.boxes_out}")

    # Which side each track is on. Which one is "home" is arbitrary; the viewer
    # only uses it to pick a colour.
    teams = {}
    shirt_store = ext_palette if ext_tracks is not None else tracker.palette
    palette = {tid: np.mean(shirt_store[tid], axis=0)
               for tid in live if shirt_store.get(tid)}

    if args.team == "zero-shot" and palette:
        # Ask the model which colour the shirt is, rather than clustering a
        # colour statistic of my own. Every track carries its mean score against
        # the colour vocabulary; the pair of words that splits them best decides
        # the two sides, so nothing is configured per fixture.
        (i, j), quality = pick_colour_pair(palette)
        for tid, s in palette.items():
            teams[tid] = "home" if s[i] > s[j] else "away"
        print(f"teams: {SHIRT_WORDS[i]} vs {SHIRT_WORDS[j]} "
              f"(separation {quality:.2f}), chosen from {len(SHIRT_WORDS)} colours")
    elif palette:
        # colour: chroma only, dropping L, which is mostly illumination and
        # whether a shirt is seen front or back.
        desc = {tid: (v / np.linalg.norm(v) if args.team == "clip" and np.linalg.norm(v)
                      else (v if args.team == "clip" else v[1:]))
                for tid, v in palette.items()}
        ids = sorted(desc)
        lab, _ = two_means([desc[i] for i in ids])
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
    # Hand-tagged events and hand-labelled identities are human work living in
    # the output file; regenerating the tracking must not silently destroy them.
    # Track ids only survive a rerun if the tracker and its inputs are unchanged,
    # so carry them forward with a warning rather than pretending it is safe.
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("events"):
            doc["events"] = old["events"]
            doc["source"] = old.get("source", doc["source"])
            print(f"kept {len(old['events'])} hand-tagged events from the previous output;")
            print("  their track ids are only valid if the tracker configuration is unchanged")
        names = {p["id"]: p for p in old.get("players", []) if p.get("name")}
        if names:
            for pl in doc["players"]:
                if pl["id"] in names:
                    pl["name"] = names[pl["id"]]["name"]
                    pl["number"] = names[pl["id"]]["number"]
            print(f"kept {len(names)} hand-labelled identities")
    out.write_text(json.dumps(doc), encoding="utf-8")

    per_frame = np.array([len(f["positions"]) for f in frames_out])
    print(f"\nwrote {out}")
    print(f"  {len(frames_out)} frames, {len(players)} tracks surviving the {args.min_samples}-sample floor")
    print(f"  players on court per frame: median {np.median(per_frame):.1f}  "
          f"min {per_frame.min()}  max {per_frame.max()}   (10 is right)")
    print(f"  dropped {dropped_class} non-player detections (referee, ball)")
    print(f"  dropped {dropped_score} player boxes below the {args.threshold} score threshold")
    print(f"  dropped {dropped_off_court} detections whose feet landed off the court")
    if dropped_unmatched:
        print(f"  dropped {dropped_unmatched} detections no tracked box agreed existed")
    lifetimes = np.array([counts[t] for t in live])
    print(f"  identity holds {np.median(lifetimes) / hz:.1f}s before a track breaks or switches")
    print(f"\nthe {args.margin:.1f}m margin is absorbing calibration error, not sideline scramble:")
    print("tighten it to 0.5m and the per-frame count falls from 9 to 7. That is the")
    print("measurement -- players plainly on the court are landing off it.")
    print("Closing the identity gap needs appearance re-identification, a different problem")
    print("from geometry, and it is not attempted here.")


if __name__ == "__main__":
    main()
