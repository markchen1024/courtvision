"""Click court landmarks on a frame to solve the court homography.

The off-the-shelf keypoint model finds 0-2 of its 33 points on our footage, which
is not enough to solve anything (a homography needs four). So the calibration
source becomes a swappable component: here it is a human clicking, in production
it is a model. The solver downstream does not care which.

Three uses, one tool:

  1. reference frame  - one calibrated frame is all a panning camera needs; every
                        other frame registers back to it by feature matching and
                        the two homographies compose.
  2. ground truth     — with hand-placed landmarks we can finally say how far off
                        a keypoint model is, in pixels, instead of eyeballing it.
  3. training labels  — the same clicks export as YOLO-pose annotations, which is
                        the fine-tuning path if the off-the-shelf model stays bad.

    python pipeline/calibrate.py --video web/media/game.mp4 --frame 120
    python pipeline/calibrate.py --video web/media/game.mp4 --every 150 --limit 30
    python pipeline/calibrate.py --check out/calibration/frame_000120.json
    python pipeline/calibrate.py --export-yolo out/court_dataset

Points or lines, whichever the frame gives you. A landmark means finding the one
pixel where two faint lines cross; a line means clicking twice anywhere along
something you can plainly see. They are worth the same to the solver -- two
constraints each -- and a line is both easier to place accurately and worth far
more coverage, because it vouches for its whole length instead of one spot.

Controls: left click places the highlighted target, or grabs a placed one if you
click within a few pixels of it. A line takes two clicks. Arrow keys nudge
whatever is grabbed, [ ] change the nudge step, E toggles contrast boost in the
loupe, N/P step targets, U undo, SPACE skip, ENTER save, Q quit.

The loupe boosts local contrast because the lines here are white paint on pale
varnished pine, a few levels of luminance apart, and un-boosted they are simply
not visible at the accuracy the homography needs.

What to look for in the output: once four points are down the court model is drawn
back over the frame. If those lines sit on the painted lines, the homography is
right. If they drift at the far end, add a landmark over there — accuracy is worst
where you have no points.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import homography

# FIBA, in metres. Big V plays FIBA, and the numbers differ from NBA enough to
# matter: the key is rectangular here, and the arc is 6.75 rather than 7.24.
COURT_LENGTH = 28.0
COURT_WIDTH = 15.0
KEY_WIDTH = 4.9
KEY_DEPTH = 5.8
CIRCLE_R = 1.8
THREE_R = 6.75
THREE_INSET = 0.90
BASKET_INSET = 1.575
COURT_AREA = COURT_LENGTH * COURT_WIDTH

# A homography is only trustworthy inside the hull of the points that made it.
# Below roughly a tenth of the court, it is fitting a patch and guessing the rest.
MIN_COVERAGE = 40.0

# Above this the points disagree with each other, not merely with the pixel grid.
MAX_RMS = 8.0

# Only line intersections and corners — anything on a smooth curve cannot be
# clicked repeatably, and a landmark you cannot place twice is worse than none.
# Order is fixed: the YOLO-pose export indexes into it.
# w/e = west/east baseline (x=0 / x=28), s/n = south/north sideline (y=0 / y=15).
LANDMARKS = [
    ("corner_ws", 0.0, 0.0),
    ("corner_wn", 0.0, COURT_WIDTH),
    ("corner_es", COURT_LENGTH, 0.0),
    ("corner_en", COURT_LENGTH, COURT_WIDTH),
    ("half_s", COURT_LENGTH / 2, 0.0),
    ("half_n", COURT_LENGTH / 2, COURT_WIDTH),
    ("circle_s", COURT_LENGTH / 2, COURT_WIDTH / 2 - CIRCLE_R),
    ("circle_n", COURT_LENGTH / 2, COURT_WIDTH / 2 + CIRCLE_R),
    ("wkey_base_s", 0.0, COURT_WIDTH / 2 - KEY_WIDTH / 2),
    ("wkey_base_n", 0.0, COURT_WIDTH / 2 + KEY_WIDTH / 2),
    ("wkey_ft_s", KEY_DEPTH, COURT_WIDTH / 2 - KEY_WIDTH / 2),
    ("wkey_ft_n", KEY_DEPTH, COURT_WIDTH / 2 + KEY_WIDTH / 2),
    ("ekey_base_s", COURT_LENGTH, COURT_WIDTH / 2 - KEY_WIDTH / 2),
    ("ekey_base_n", COURT_LENGTH, COURT_WIDTH / 2 + KEY_WIDTH / 2),
    ("ekey_ft_s", COURT_LENGTH - KEY_DEPTH, COURT_WIDTH / 2 - KEY_WIDTH / 2),
    ("ekey_ft_n", COURT_LENGTH - KEY_DEPTH, COURT_WIDTH / 2 + KEY_WIDTH / 2),
    ("w3pt_s", 0.0, THREE_INSET),
    ("w3pt_n", 0.0, COURT_WIDTH - THREE_INSET),
    ("e3pt_s", COURT_LENGTH, THREE_INSET),
    ("e3pt_n", COURT_LENGTH, COURT_WIDTH - THREE_INSET),
]
BY_NAME = {n: (x, y) for n, x, y in LANDMARKS}

# Lines you can name at a glance, each stored as the court segment that defines
# it. Two clicks anywhere along one in the image is a full correspondence -- the
# same two constraints a landmark gives, without having to find an intersection.
# key_side_* is one infinite line through both keys, so click it at whichever end
# is in shot.
LINES = [
    ("baseline_w", ((0.0, 0.0), (0.0, COURT_WIDTH))),
    ("baseline_e", ((COURT_LENGTH, 0.0), (COURT_LENGTH, COURT_WIDTH))),
    ("sideline_s", ((0.0, 0.0), (COURT_LENGTH, 0.0))),
    ("sideline_n", ((0.0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH))),
    ("halfway", ((COURT_LENGTH / 2, 0.0), (COURT_LENGTH / 2, COURT_WIDTH))),
    ("ft_w", ((KEY_DEPTH, COURT_WIDTH / 2 - KEY_WIDTH / 2), (KEY_DEPTH, COURT_WIDTH / 2 + KEY_WIDTH / 2))),
    ("ft_e", ((COURT_LENGTH - KEY_DEPTH, COURT_WIDTH / 2 - KEY_WIDTH / 2),
              (COURT_LENGTH - KEY_DEPTH, COURT_WIDTH / 2 + KEY_WIDTH / 2))),
    ("key_side_s", ((0.0, COURT_WIDTH / 2 - KEY_WIDTH / 2), (KEY_DEPTH, COURT_WIDTH / 2 - KEY_WIDTH / 2))),
    ("key_side_n", ((0.0, COURT_WIDTH / 2 + KEY_WIDTH / 2), (KEY_DEPTH, COURT_WIDTH / 2 + KEY_WIDTH / 2))),
]
LINE_BY_NAME = dict(LINES)

# One flat list so N/P walks landmarks and lines alike.
TARGETS = ([("point", n) for n, _, _ in LANDMARKS] + [("line", n) for n, _ in LINES])

# Blueprint palette, BGR, matching try_models.py and the viewer
LIME = (49, 240, 200)
RED = (107, 114, 244)
TEAL = (168, 201, 59)
DIM = (111, 162, 154)
INK = (24, 22, 20)

# waitKeyEx codes: Windows first, then the Qt/GTK builds.
ARROWS = {2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1),
          81: (-1, 0), 83: (1, 0), 82: (0, -1), 84: (0, 1)}


def three_point_line(west):
    """Straight sections plus the arc, as one polyline."""
    basket_x = BASKET_INSET if west else COURT_LENGTH - BASKET_INSET
    base_x = 0.0 if west else COURT_LENGTH
    sign = 1.0 if west else -1.0

    dy = COURT_WIDTH / 2 - THREE_INSET
    dx = float(np.sqrt(THREE_R ** 2 - dy ** 2))  # where the arc meets the straight
    tan_x = basket_x + sign * dx
    half = float(np.degrees(np.arctan2(dy, dx)))

    # Sweep so the arc starts on the south side, matching the point order below.
    span = (-half, half) if west else (180 + half, 180 - half)
    th = np.radians(np.linspace(span[0], span[1], 64))
    arc = list(zip(basket_x + THREE_R * np.cos(th), COURT_WIDTH / 2 + THREE_R * np.sin(th)))

    return (
        [(base_x, THREE_INSET), (tan_x, THREE_INSET)]
        + arc
        + [(tan_x, COURT_WIDTH - THREE_INSET), (base_x, COURT_WIDTH - THREE_INSET)]
    )


def court_polylines():
    """The court as polylines in metres — drawn over the frame to check the fit."""
    L, W = COURT_LENGTH, COURT_WIDTH
    t = np.linspace(0, 2 * np.pi, 96)
    lines = [
        [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],
        [(L / 2, 0), (L / 2, W)],
        list(zip(L / 2 + CIRCLE_R * np.cos(t), W / 2 + CIRCLE_R * np.sin(t))),
    ]
    for west in (True, False):
        base_x = 0.0 if west else L
        ft_x = KEY_DEPTH if west else L - KEY_DEPTH
        lines.append([
            (base_x, W / 2 - KEY_WIDTH / 2), (ft_x, W / 2 - KEY_WIDTH / 2),
            (ft_x, W / 2 + KEY_WIDTH / 2), (base_x, W / 2 + KEY_WIDTH / 2),
        ])
        lines.append(list(zip(ft_x + CIRCLE_R * np.cos(t), W / 2 + CIRCLE_R * np.sin(t))))
        lines.append(three_point_line(west))
    return [np.array(p, np.float32) for p in lines]


def mirror_index():
    """YOLO's flip_idx: where each landmark lands when the image is flipped.

    The camera sits outside a sideline, so left-right in the image runs along the
    court's length. Flipping horizontally therefore swaps the two ends (west<->east)
    and leaves the sidelines alone — halfway and centre-circle points map to
    themselves. Derived from the coordinates rather than the names, because getting
    this backwards trains the model on silently mislabelled data.
    """
    idx = []
    for name, x, y in LANDMARKS:
        partner = next(
            (k for k, (_, mx, my) in enumerate(LANDMARKS)
             if abs(mx - (COURT_LENGTH - x)) < 1e-6 and abs(my - y) < 1e-6),
            None,
        )
        if partner is None:
            raise SystemExit(f"landmark {name} has no mirror partner; flip_idx would be wrong")
        idx.append(partner)
    return idx


def solve(picked, lines=None):
    """Return (H court->image, rms px, spread in m^2, residuals per name).

    Points and lines are interchangeable: each contributes two constraints, so
    four of anything is the minimum. Clicking a line means clicking twice
    anywhere along something you can see, rather than once on an intersection you
    have to find, which is the accuracy this footage does not otherwise give.

    residuals maps name -> pixels of disagreement, which is the only thing that
    says *which* click was wrong. With exactly four correspondences the fit is
    always exact, so rms stays at zero and says nothing; the fifth is the first
    real check.
    """
    lines = lines or {}
    point_pairs = [(BY_NAME[n], picked[n]) for n in picked]
    line_pairs = [(LINE_BY_NAME[n], lines[n]) for n in lines if n in LINE_BY_NAME]
    if 2 * (len(point_pairs) + len(line_pairs)) < 8:
        return None, None, None, {}

    # A line vouches for the stretch of court it runs along, so its defining
    # segment counts toward coverage the same way a clicked landmark does.
    hull_src = ([BY_NAME[n] for n in picked]
                + [p for n in line_pairs for p in n[0]])
    spread = float(cv2.contourArea(cv2.convexHull(np.array(hull_src, np.float32))))

    H = homography.solve(point_pairs, line_pairs)
    if H is None:
        return None, None, spread, {}

    r = homography.residuals(H, point_pairs, line_pairs)
    res = {}
    for n, e in zip(picked, r["points"]):
        res[n] = float(e)
    for (n, _), e in zip([(n, None) for n in lines if n in LINE_BY_NAME], r["lines"]):
        res[n] = float(e)
    err = np.array(list(res.values()), np.float64)
    return H, float(np.sqrt((err ** 2).mean())), spread, res


def draw_court(img, H, colour=TEAL, thickness=2):
    if H is None:
        return img
    for poly in court_polylines():
        pts = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H).reshape(-1, 2)
        if not np.isfinite(pts).all():
            continue
        pts = np.clip(pts, -1e4, 1e4).astype(np.int32)
        cv2.polylines(img, [pts], False, colour, thickness, cv2.LINE_AA)
    return img


def enhance(bgr):
    """Pull the court lines out of the floor.

    White paint on pale varnished pine is a few levels of luminance apart, and the
    lines are the whole job here. CLAHE on L alone stretches that gap locally
    without touching hue, which keeps the blue key looking like the blue key.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def magnifier(frame, cx, cy, size=90, zoom=5, boost=True):
    """Line intersections need pixel-accurate clicks; a fit-to-screen view cannot give that."""
    h, w = frame.shape[:2]
    half = size // 2
    x0, y0 = int(cx) - half, int(cy) - half
    crop = np.zeros((size, size, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    if boost:
        crop = enhance(crop)
    # Cubic, not nearest: at 5x the nearest-neighbour blocks are themselves the
    # thing you end up aiming at, which is worse than a slightly soft edge.
    big = cv2.resize(crop, (size * zoom, size * zoom), interpolation=cv2.INTER_CUBIC)
    c = size * zoom // 2
    cv2.line(big, (c, 0), (c, c - 9), LIME, 1)
    cv2.line(big, (c, c + 9), (c, size * zoom), LIME, 1)
    cv2.line(big, (0, c), (c - 9, c), LIME, 1)
    cv2.line(big, (c + 9, c), (size * zoom, c), LIME, 1)
    cv2.circle(big, (c, c), 2, RED, -1)
    cv2.rectangle(big, (0, 0), (size * zoom - 1, size * zoom - 1), LIME, 2)
    return big


def diagram(target_idx, picked, lines=None, width=430):
    """Top-down court with the current target ringed — names alone are ambiguous
    when you are staring at a camera view and cannot tell north from south."""
    scale = (width - 24) / COURT_LENGTH
    height = int(COURT_WIDTH * scale) + 24
    img = np.full((height, width, 3), 28, np.uint8)

    def P(x, y):
        return (int(12 + x * scale), int(12 + (COURT_WIDTH - y) * scale))

    for poly in court_polylines():
        cv2.polylines(img, [np.array([P(x, y) for x, y in poly], np.int32)], False, DIM, 1, cv2.LINE_AA)
    for name, (a, b) in LINES:
        if name in (lines or {}):
            cv2.line(img, P(*a), P(*b), LIME, 2, cv2.LINE_AA)
    for name, x, y in LANDMARKS:
        cv2.circle(img, P(x, y), 3, LIME if name in picked else (80, 80, 80), -1)

    kind, name = TARGETS[target_idx]
    if kind == "point":
        x, y = BY_NAME[name]
        cv2.circle(img, P(x, y), 9, RED, 2, cv2.LINE_AA)
    else:
        a, b = LINE_BY_NAME[name]
        cv2.line(img, P(*a), P(*b), RED, 3, cv2.LINE_AA)
    return img


def banner(canvas, lines):
    pad = 8
    h = 20 * len(lines) + pad
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], h), INK, -1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(canvas, text, (pad, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return canvas


def calibrate_frame(frame, frame_idx, display_width):
    """Interactive loop for one frame. Returns picked dict, or None if quit."""
    h, w = frame.shape[:2]
    scale = min(1.0, display_width / w)
    disp_size = (int(w * scale), int(h * scale))

    state = {"cursor": (w // 2, h // 2), "click": None, "moved": True}

    def on_mouse(event, x, y, flags, _):
        full = (x / scale, y / scale)
        if event == cv2.EVENT_MOUSEMOVE:
            state["cursor"] = full
            state["moved"] = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = full

    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    picked, lines, target, confirm = {}, {}, 0, False
    active, boost, step, pending = None, True, 1.0, None

    def grabbable():
        """Every clicked pixel that can be picked up again: landmarks, and the two
        ends of each placed line."""
        out = [(n, p, None) for n, p in picked.items()]
        out += [(n, p, e) for n, seg in lines.items() for e, p in enumerate(seg)]
        return out

    while True:
        if state["click"] is not None:
            cx, cy = state["click"]
            near = [(n, e) for n, p, e in grabbable()
                    if abs(p[0] - cx) < 10 / scale and abs(p[1] - cy) < 10 / scale]
            kind, name = TARGETS[target]
            if near and pending is None:
                # Clicking near something already placed grabs it instead of
                # dropping a new one, so a click 3px out gets nudged, not redone.
                active, grabbed_end = near[0]
                pending = None
                target = [n for _, n in TARGETS].index(active)
                state["end"] = grabbed_end
            elif kind == "point":
                picked[name] = (cx, cy)
                active, state["end"] = name, None
                target = (target + 1) % len(TARGETS)
            else:
                # A line takes two clicks, anywhere along it. They do not have to
                # be the ends of anything -- that is the whole point.
                if pending is None:
                    pending = (cx, cy)
                else:
                    lines[name] = (pending, (cx, cy))
                    pending = None
                    active, state["end"] = name, 1
                    target = (target + 1) % len(TARGETS)
            state["click"] = None

        H, rms, spread, residuals = solve(picked, lines)
        worst = max(residuals, key=residuals.get) if residuals else None

        view = cv2.resize(frame, disp_size) if scale < 1.0 else frame.copy()
        if H is not None:
            # Scale the homography into display space rather than drawing at full
            # res and shrinking — keeps the lines crisp.
            S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64)
            draw_court(view, S @ H)
            # Show where the fit thinks each point should have gone. The gap is
            # the mistake, drawn rather than described.
            for name, (px, py) in picked.items():
                want = cv2.perspectiveTransform(
                    np.array([[BY_NAME[name]]], np.float32), S @ H).reshape(2)
                if np.isfinite(want).all() and residuals.get(name, 0) > 4.0:
                    cv2.line(view, (int(px * scale), int(py * scale)),
                             (int(want[0]), int(want[1])), RED, 1, cv2.LINE_AA)
        # Placed lines, drawn out to the frame edges so you can see whether they
        # follow the paint far away from where you happened to click.
        for name, ((ax, ay), (bx, by)) in lines.items():
            bad = name == worst and residuals.get(name, 0) > 4.0
            colour = RED if bad else LIME
            a = np.array([ax * scale, ay * scale])
            b = np.array([bx * scale, by * scale])
            d_ = b - a
            n_ = np.linalg.norm(d_)
            if n_ > 1e-6:
                far = 4000.0 / n_ * d_
                cv2.line(view, tuple((a - far).astype(int)), tuple((b + far).astype(int)),
                         colour, 1, cv2.LINE_AA)
            for e, p in enumerate(((ax, ay), (bx, by))):
                q = (int(p[0] * scale), int(p[1] * scale))
                cv2.circle(view, q, 4, colour, -1)
                if name == active and state.get("end") == e:
                    cv2.circle(view, q, 9, (255, 255, 255), 1, cv2.LINE_AA)
            tag = name + (f" {residuals[name]:.0f}px" if name in residuals else "")
            cv2.putText(view, tag, (int(ax * scale) + 8, int(ay * scale) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
        if pending is not None:
            q = (int(pending[0] * scale), int(pending[1] * scale))
            cv2.circle(view, q, 4, RED, -1)
            cv2.line(view, q, (int(state["cursor"][0] * scale), int(state["cursor"][1] * scale)),
                     RED, 1, cv2.LINE_AA)

        for name, (px, py) in picked.items():
            p = (int(px * scale), int(py * scale))
            bad = name == worst and residuals.get(name, 0) > 4.0
            cv2.circle(view, p, 5, RED if bad else LIME, -1)
            cv2.circle(view, p, 5, INK, 1)
            if bad:
                cv2.circle(view, p, 12, RED, 2, cv2.LINE_AA)
            if name == active:
                cv2.circle(view, p, 9, (255, 255, 255), 1, cv2.LINE_AA)
            tag = name + (f" {residuals[name]:.0f}px" if name in residuals else "")
            cv2.putText(view, tag, (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        RED if bad else LIME, 1, cv2.LINE_AA)

        d = diagram(target, picked, lines)
        view[view.shape[0] - d.shape[0]:, :d.shape[1]] = d
        # While nudging, the loupe follows the point being nudged; otherwise the
        # cursor. Watching your own arrow keys move the crosshair is the point.
        at = state["cursor"]
        if not state["moved"]:
            if active in picked:
                at = picked[active]
            elif active in lines and state.get("end") is not None:
                at = lines[active][state["end"]]
        m = magnifier(frame, *at, boost=boost)
        view[view.shape[0] - m.shape[0]:, view.shape[1] - m.shape[1]:] = m

        kind, tname = TARGETS[target]
        need = 4 - len(picked) - len(lines)
        status = [(f"frame {frame_idx}   target: {tname} ({kind})"
                   + ("  <- click twice anywhere along it" if kind == "line" else "")
                   + f"   placed {len(picked)}pt {len(lines)}line", LIME)]
        good = False
        if H is None and need > 0:
            status.append((f"need {need} more point(s) or line(s) - either counts the same", DIM))
        elif H is None:
            status.append(("no unique fit - parallel lines only, or points sitting on their own lines",
                           RED))
        else:
            good = rms <= MAX_RMS and spread >= MIN_COVERAGE
            status.append((f"reprojection {rms:.1f}px   coverage {spread:.0f}m2 "
                           f"({spread / COURT_AREA * 100:.0f}% of court)", LIME if good else RED))
            if len(picked) + len(lines) == 4:
                status.append(("four always fit exactly - add a fifth to check yourself", DIM))
            elif rms > MAX_RMS:
                status.append((f"worst: {worst} is {residuals[worst]:.0f}px out - press U on it and re-click",
                               RED))
            if spread < MIN_COVERAGE:
                status.append(("thin coverage - the fit will drift away from your points", RED))
        status.append((f"arrows nudge {active or '-'} by {step:g}px  |  [ ] step  E contrast {'on' if boost else 'off'}",
                       DIM))
        if confirm:
            status.append(("ENTER again to save it anyway, any other key to keep fixing", RED))
        banner(view, status)

        cv2.imshow(win, view)
        raw = cv2.waitKeyEx(20)
        if raw in ARROWS and (active in picked or active in lines):
            dx, dy = ARROWS[raw]
            if active in picked:
                px, py = picked[active]
                picked[active] = (px + dx * step, py + dy * step)
            else:
                e = state.get("end") or 0
                seg = list(lines[active])
                seg[e] = (seg[e][0] + dx * step, seg[e][1] + dy * step)
                lines[active] = tuple(seg)
            state["moved"] = False
            continue
        key = raw & 0xFF if raw != -1 else 255
        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            return None, None
        if key in (13, 10):
            # A bad fit used to save silently, which is how a 95px calibration ends
            # up downstream still looking like a finished measurement.
            if H is not None and not good and not confirm:
                confirm = True
            else:
                cv2.destroyWindow(win)
                return picked, lines
            continue
        if key != 255:
            confirm = False
        if key == ord("e"):
            boost = not boost
        elif key == ord("["):
            step = max(0.25, step / 2)
        elif key == ord("]"):
            step = min(8.0, step * 2)
        elif key == ord("n") or key == ord(" "):
            target, pending = (target + 1) % len(TARGETS), None
        elif key == ord("p"):
            target, pending = (target - 1) % len(TARGETS), None
        elif key == ord("u"):
            if pending is not None:
                pending = None
            else:
                doomed = active or TARGETS[target][1]
                picked.pop(doomed, None)
                lines.pop(doomed, None)
                active = None


def save_calibration(path, video, frame_idx, shape, picked, H, rms, spread, residuals=None,
                     lines=None):
    Hinv = np.linalg.inv(H) if H is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "video": str(video),
        "frame": frame_idx,
        "width": shape[1],
        "height": shape[0],
        "court": "FIBA_28x15",
        "points": [
            {"name": n, "court": list(BY_NAME[n]), "pixel": list(p),
             "residual_px": (residuals or {}).get(n)}
            for n, p in picked.items()
        ],
        "lines": [
            {"name": n, "court": [list(a), list(b)], "pixel": [list(seg[0]), list(seg[1])],
             "residual_px": (residuals or {}).get(n)}
            for n, seg in (lines or {}).items()
            for (a, b) in [LINE_BY_NAME[n]]
        ],
        "partial": H is None,
        "H_court_to_image": H.tolist() if H is not None else None,
        "H_image_to_court": Hinv.tolist() if Hinv is not None else None,
        "reprojection_rms_px": rms,
        "coverage_m2": spread,
    }, indent=2))


def cmd_calibrate(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.every:
        targets = list(range(0, total, args.every))[:args.limit]
    else:
        targets = [args.frame]
    print(f"{args.video}: {total} frames - calibrating {len(targets)}: {targets[:8]}{'...' if len(targets) > 8 else ''}")

    outdir = Path(args.out)
    done = 0
    for idx in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {idx}: unreadable, skipping")
            continue

        picked, drawn = calibrate_frame(frame, idx, args.display_width)
        if picked is None:
            print("  quit")
            break
        H, rms, spread, residuals = solve(picked, drawn)
        if H is None:
            # A side-on view of one end cannot always spread four points. Keep the
            # clicks anyway: fuse_calibration.py transports points from several
            # frames into one and solves them together, so a frame that shows only
            # the centre circle still contributes the two points it does have.
            if args.partial and (picked or drawn):
                path = outdir / f"frame_{idx:06d}.json"
                save_calibration(path, args.video, idx, frame.shape, picked, None, None, None,
                                 residuals, drawn)
                print(f"  frame {idx}: {len(picked)}pt {len(drawn)}line saved as partial "
                      f"(no homography on its own)")
                done += 1
            else:
                print(f"  frame {idx}: {len(picked)}pt {len(drawn)}line is not enough "
                      f"(use --partial to keep them)")
            continue

        path = outdir / f"frame_{idx:06d}.json"
        save_calibration(path, args.video, idx, frame.shape, picked, H, rms, spread, residuals, drawn)
        overlay = draw_court(frame.copy(), H)
        cv2.imwrite(str(outdir / f"frame_{idx:06d}.jpg"), overlay)
        print(f"  frame {idx}: {len(picked)}pt {len(drawn)}line, rms {rms:.1f}px, "
              f"coverage {spread:.0f}m2 -> {path.name}")
        done += 1

    cap.release()
    print(f"\nsaved {done} calibration(s) to {outdir}")
    if done:
        print("check the .jpg overlays: the drawn court should sit on the painted lines")


def cmd_check(args):
    data = json.loads(Path(args.check).read_text())
    cap = cv2.VideoCapture(data["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, data["frame"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {data['frame']} of {data['video']}")

    H = np.array(data["H_court_to_image"], np.float64)
    out = draw_court(frame, H)
    for p in data["points"]:
        cv2.circle(out, (int(p["pixel"][0]), int(p["pixel"][1])), 6, LIME, -1)
    path = Path(args.check).with_suffix(".check.jpg")
    cv2.imwrite(str(path), out)
    print(f"frame {data['frame']}: {len(data['points'])}pt {len(data.get('lines', []))}line, "
          f"rms {data['reprojection_rms_px']:.1f}px, coverage {data['coverage_m2']:.0f}m2")
    if data["reprojection_rms_px"] > MAX_RMS:
        print(f"  FAILED: above {MAX_RMS:.0f}px the points contradict each other, "
              f"so the court drawn in the overlay is not where the court is.")
    # Recomputed rather than read back, so an older file without residuals still
    # gets diagnosed.
    _, _, _, residuals = solve(
        {p["name"]: tuple(p["pixel"]) for p in data["points"]},
        {l["name"]: (tuple(l["pixel"][0]), tuple(l["pixel"][1])) for l in data.get("lines", [])})
    for name, err in sorted(residuals.items(), key=lambda kv: -kv[1]):
        flag = "  <-- re-click this one" if err > MAX_RMS else ""
        print(f"    {name:14s} {err:7.1f}px{flag}")
    print(f"wrote {path}")


def cmd_export_yolo(args):
    """Write the clicks as a YOLO-pose dataset — the fine-tuning path.

    The tracked object is the court itself, so the box is the extent of the
    visible landmarks. Landmarks that were not clicked get visibility 0.
    """
    src = Path(args.out)
    files = sorted(src.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"no calibrations in {src} - run the calibrate mode first")

    dst = Path(args.export_yolo)
    (dst / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "train").mkdir(parents=True, exist_ok=True)

    caps = {}
    written = 0
    for f in files:
        data = json.loads(f.read_text())
        video = data["video"]
        if video not in caps:
            caps[video] = cv2.VideoCapture(video)
        cap = caps[video]
        cap.set(cv2.CAP_PROP_POS_FRAMES, data["frame"])
        ok, frame = cap.read()
        if not ok:
            continue

        W, H_px = data["width"], data["height"]
        by_name = {p["name"]: p["pixel"] for p in data["points"]}
        xs = [p[0] for p in by_name.values()]
        ys = [p[1] for p in by_name.values()]
        x0, x1 = max(0.0, min(xs) - 20), min(float(W), max(xs) + 20)
        y0, y1 = max(0.0, min(ys) - 20), min(float(H_px), max(ys) + 20)

        fields = ["0",
                  f"{(x0 + x1) / 2 / W:.6f}", f"{(y0 + y1) / 2 / H_px:.6f}",
                  f"{(x1 - x0) / W:.6f}", f"{(y1 - y0) / H_px:.6f}"]
        for name, _, _ in LANDMARKS:
            if name in by_name:
                px, py = by_name[name]
                fields += [f"{px / W:.6f}", f"{py / H_px:.6f}", "2"]
            else:
                fields += ["0.000000", "0.000000", "0"]

        stem = f"{Path(video).stem}_{data['frame']:06d}"
        cv2.imwrite(str(dst / "images" / "train" / f"{stem}.jpg"), frame)
        (dst / "labels" / "train" / f"{stem}.txt").write_text(" ".join(fields) + "\n")
        written += 1

    for cap in caps.values():
        cap.release()

    flip = mirror_index()

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        f"kpt_shape: [{len(LANDMARKS)}, 3]\n"
        f"flip_idx: {flip}\n"
        "names:\n  0: court\n"
    )
    print(f"wrote {written} image/label pairs to {dst}")
    print(f"train with: yolo pose train data={dst / 'data.yaml'} model=yolo11n-pose.pt epochs=100")
    if written < 30:
        print(f"note: {written} frames is thin for fine-tuning - 30-50 is the usual floor")


def cmd_selftest():
    """Verify the geometry without needing a display: invent a camera, project the
    landmarks through it, solve back, and check we recover the same homography."""
    src = np.array([[x, y] for _, x, y in LANDMARKS], np.float32)
    truth = np.array([[42.0, -6.0, 300.0], [3.0, 28.0, 120.0], [0.0018, -0.0009, 1.0]])
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), truth).reshape(-1, 2)

    for n in (4, 6, 20):
        picked = {LANDMARKS[i][0]: tuple(proj[i]) for i in np.linspace(0, 19, n).astype(int)}
        H, rms, spread, _ = solve(picked)
        assert H is not None, f"{n} points failed to solve"
        print(f"  {n:2d} points: rms {rms:.4f}px  coverage {spread:.0f}m2  {'ok' if rms < 0.5 else 'FAIL'}")
        assert rms < 0.5

    # Two bad picks that must not sail through. Everything on one baseline has no
    # solution at all; one key rectangle solves perfectly and still extrapolates
    # garbage to the far end, so only coverage catches it.
    line = {n: tuple(proj[i]) for i, (n, x, _) in enumerate(LANDMARKS) if x == 0.0}
    H, _, spread, _ = solve(line)
    print(f"  collinear pick ({len(line)} pts): coverage {spread:.0f}m2 -> "
          f"{'rejected' if H is None else 'ACCEPTED, BAD'}")
    assert H is None

    patch = {n: tuple(proj[i]) for i, (n, _, _) in enumerate(LANDMARKS) if n.startswith("wkey")}
    H, rms, spread, _ = solve(patch)
    print(f"  one-key pick ({len(patch)} pts): rms {rms:.4f}px coverage {spread:.0f}m2 -> "
          f"{'flagged' if spread < MIN_COVERAGE else 'NOT FLAGGED'}")
    assert H is not None and rms < 0.5 and spread < MIN_COVERAGE

    assert len(court_polylines()) == 9
    for west in (True, False):
        tp = np.array(three_point_line(west))
        basket = np.array([BASKET_INSET if west else COURT_LENGTH - BASKET_INSET, COURT_WIDTH / 2])
        arc = tp[2:-2]
        radii = np.linalg.norm(arc - basket, axis=1)
        assert np.allclose(radii, THREE_R, atol=1e-9), f"arc is not {THREE_R}m from the basket"
        # The straights have to meet the arc, or the line has a visible kink.
        assert np.allclose(tp[1], arc[0], atol=1e-9) and np.allclose(tp[-2], arc[-1], atol=1e-9)
        # ...and start on the baseline, inset from the sideline.
        assert abs(tp[0][0] - (0.0 if west else COURT_LENGTH)) < 1e-9
        assert abs(tp[0][1] - THREE_INSET) < 1e-9
        print(f"  3pt line {'west' if west else 'east'}: radius {radii.mean():.3f}m, "
              f"straights meet the arc")
    flip = mirror_index()
    assert [flip[i] for i in flip] == list(range(len(LANDMARKS))), "flip_idx is not an involution"
    assert flip[LANDMARKS.index(("corner_ws", 0.0, 0.0))] == [n for n, _, _ in LANDMARKS].index("corner_es")
    assert flip[[n for n, _, _ in LANDMARKS].index("half_s")] == [n for n, _, _ in LANDMARKS].index("half_s")
    print(f"  flip_idx ok (west<->east, {sum(1 for i, j in enumerate(flip) if i == j)} self-mapped)")
    print("  court geometry ok")
    print("\nselftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="web/media/game.mp4")
    ap.add_argument("--frame", type=int, default=0, help="which frame to calibrate")
    ap.add_argument("--every", type=int, help="calibrate one frame in N (batch labelling)")
    ap.add_argument("--limit", type=int, default=40, help="max frames in batch mode")
    ap.add_argument("--out", default="out/calibration")
    ap.add_argument("--display-width", type=int, default=1500)
    ap.add_argument("--check", help="render a saved calibration and exit")
    ap.add_argument("--export-yolo", help="write --out as a YOLO-pose dataset here")
    ap.add_argument("--partial", action="store_true",
                    help="keep clicks even when they cannot solve alone (for fuse_calibration.py)")
    ap.add_argument("--selftest", action="store_true", help="verify the geometry, no display needed")
    args = ap.parse_args()

    if args.selftest:
        cmd_selftest()
    elif args.check:
        cmd_check(args)
    elif args.export_yolo:
        cmd_export_yolo(args)
    else:
        cmd_calibrate(args)


if __name__ == "__main__":
    main()
