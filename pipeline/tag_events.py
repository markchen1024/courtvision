"""Tag events by hand, and let the geometry supply everything it can.

Event recognition is the part Superstat has spent two years on. A weekend version
would be worse and would invite exactly the comparison worth avoiding, so the
events here are tagged by a person and the viewer says so.

What is not tagged by hand is anything already measured. Click the player, press
a key for what happened, and the tool reads the shot location out of the tracking
data at that moment. So the shot chart is made of positions the pipeline
computed, not positions someone drew, and whether a shot is worth two or three is
decided by where the player was standing rather than by the tagger's opinion.

    python pipeline/tag_events.py --data web/data/nba.json

  SPACE  play / pause          M  made two            R  rebound
  , .    step one sample       3  made three          A  assist
  [ ]    jump five seconds     X  missed              T  turnover
  click  select a player       U  undo last event     ENTER  save
                               Q  quit without saving

The 2/3 line is approximate. The court model is a 28x15 box fitted to NBA courts,
which are 28.65m, so distances carry about 2% of scale error -- 15cm at the arc.
Fine everywhere except for a shot taken right on the line.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# NBA: 23'9" to the arc, basket centre 63" off the baseline. Both expressed in
# the court model's coordinates, which is where the 2% comes from.
THREE_R = 7.24
BASKET_INSET = 1.60
CORNER_INSET = 0.914     # 3 feet, where the corner segments sit off each sideline

LIME = (49, 240, 200)
RED = (107, 114, 244)
TEAL = (168, 201, 59)
DIM = (111, 162, 154)
INK = (24, 22, 20)

KEYS = {
    ord("m"): ("shot_made", 2),
    ord("3"): ("shot_made", 3),
    ord("x"): ("shot_missed", None),
    ord("r"): ("rebound", None),
    ord("a"): ("assist", None),
    ord("t"): ("turnover", None),
}


def shot_value(x, y, court_length, court_width):
    """Two or three, from where the player was standing.

    The three-point line is not a circle. The arc is 23'9" from the basket, but
    in the corners it is cut to two straight segments three feet in from each
    sideline, because there is no room for the arc. Treating it as one radius
    calls every corner three a two -- 6.90m against a 7.24m threshold -- which is
    wrong in exactly the spot teams shoot from most.

    Whichever basket is nearer is the one being shot at. With one clip and no
    possession model that is the honest assumption, and it only fails for a shot
    from the far half, which is not a shot.
    """
    basket_x = min((BASKET_INSET, court_length - BASKET_INSET),
                   key=lambda bx: abs(x - bx))
    d = float(np.hypot(x - basket_x, y - court_width / 2))

    # Where the straight segment gives way to the arc.
    dy = court_width / 2 - CORNER_INSET
    tangent = float(np.sqrt(max(0.0, THREE_R ** 2 - dy ** 2)))
    in_corner = abs(x - basket_x) < tangent and (x - basket_x) * (
        1 if basket_x < court_length / 2 else -1) < tangent

    if in_corner:
        return (3 if (y < CORNER_INSET or y > court_width - CORNER_INSET) else 2), d
    return (3 if d > THREE_R else 2), d


def load_homographies(path):
    if not Path(path).exists():
        return {}
    doc = json.loads(Path(path).read_text())
    return {r["frame"]: np.array(r["H_court_to_image"], np.float64)
            for r in doc["frames"] if r.get("solved")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="web/data/nba.json")
    ap.add_argument("--video", default="web/media/nba.mp4")
    ap.add_argument("--calibration", default="out/auto_calibration.json")
    ap.add_argument("--display-width", type=int, default=1500)
    args = ap.parse_args()

    doc = json.loads(Path(args.data).read_text())
    frames = doc["frames"]
    players = {p["id"]: p for p in doc["players"]}
    L, W = doc["court"]["length_m"], doc["court"]["width_m"]
    hz = doc["video"]["hz"]
    homographies = load_homographies(args.calibration)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = min(1.0, args.display_width / w)

    print(f"{len(frames)} tracked samples at {hz:.0f}Hz, {len(players)} tracks")
    print(f"{len(doc['events'])} events already tagged")

    events = list(doc["events"])
    i = 0
    playing = False
    selected = None
    state = {"click": None}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x / scale, y / scale)

    win = "tag events"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        sample = frames[max(0, min(i, len(frames) - 1))]
        t = sample["t"]
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        H = homographies.get(frame_idx)
        if H is None and homographies:
            near = min(homographies, key=lambda f: abs(f - frame_idx))
            H = homographies[near] if abs(near - frame_idx) <= fps / hz else None

        # Where each tracked player is, in this image
        onscreen = {}
        if H is not None:
            for pos in sample["positions"]:
                q = cv2.perspectiveTransform(
                    np.float32([[[pos["x"], pos["y"]]]]), H).reshape(2)
                if np.isfinite(q).all():
                    onscreen[pos["id"]] = (float(q[0]), float(q[1]))

        if state["click"] is not None and onscreen:
            cx, cy = state["click"]
            tid, d = min(((k, float(np.hypot(p[0] - cx, p[1] - cy)))
                          for k, p in onscreen.items()), key=lambda kv: kv[1])
            selected = tid if d < 90 else None
            state["click"] = None

        view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
        for tid, (px, py) in onscreen.items():
            p = (int(px * scale), int(py * scale))
            chosen = tid == selected
            colour = RED if chosen else (LIME if players[tid]["team"] == "home" else TEAL)
            cv2.circle(view, p, 13 if chosen else 9, colour, 2 if chosen else -1, cv2.LINE_AA)
            cv2.putText(view, str(tid), (p[0] + 12, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        recent = [e for e in events if abs(e["t"] - t) < 2.0]
        lines = [
            (f"{t:6.2f}s   sample {i + 1}/{len(frames)}   "
             f"{len(onscreen)} tracked   {'PLAYING' if playing else 'paused'}", LIME),
            (f"selected: {'T' + str(selected) if selected else 'click a player'}"
             f"    events tagged: {len(events)}", RED if selected else DIM),
            ("M made2  3 made3  X miss  R reb  A ast  T turnover  U undo  ENTER save", DIM),
        ]
        if recent:
            lines.append(("nearby: " + ", ".join(f"{e['type']}@{e['t']:.1f}" for e in recent[-3:]),
                          TEAL))
        pad = 8
        cv2.rectangle(view, (0, 0), (view.shape[1], 20 * len(lines) + pad), INK, -1)
        for n, (text, colour) in enumerate(lines):
            cv2.putText(view, text, (pad, 18 + 20 * n), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1, cv2.LINE_AA)

        cv2.imshow(win, view)
        key = cv2.waitKey(int(1000 / hz) if playing else 20) & 0xFF

        if playing and key == 255:
            i = min(i + 1, len(frames) - 1)
            continue
        if key in (ord("q"), 27):
            print("quit without saving")
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == ord(" "):
            playing = not playing
        elif key == ord(","):
            i, playing = max(0, i - 1), False
        elif key == ord("."):
            i, playing = min(len(frames) - 1, i + 1), False
        elif key == ord("["):
            i, playing = max(0, i - int(5 * hz)), False
        elif key == ord("]"):
            i, playing = min(len(frames) - 1, i + int(5 * hz)), False
        elif key == ord("u"):
            if events:
                gone = events.pop()
                print(f"  undo {gone['type']} at {gone['t']}s")
        elif key in (13, 10):
            break
        elif key in KEYS:
            if selected is None:
                print("  pick a player first")
                continue
            kind, points = KEYS[key]
            pos = next((p for p in sample["positions"] if p["id"] == selected), None)
            ev = {"t": round(t, 2), "type": kind, "player": selected,
                  "team": players[selected]["team"]}
            if kind.startswith("shot_") and pos:
                # Location and value both come from the measurement, not the tagger.
                value, dist = shot_value(pos["x"], pos["y"], L, W)
                ev.update({"x": round(pos["x"], 2), "y": round(pos["y"], 2),
                           "points": points or value})
                if points and points != value:
                    print(f"  note: you said {points}, the position says {value} "
                          f"({dist:.1f}m from the basket) -- keeping yours")
                print(f"  {kind} by T{selected} at {t:.2f}s, "
                      f"{dist:.1f}m out, worth {ev['points']}")
            else:
                print(f"  {kind} by T{selected} at {t:.2f}s")
            events.append(ev)
            playing = False

    cap.release()
    cv2.destroyAllWindows()

    events.sort(key=lambda e: e["t"])
    doc["events"] = events
    made = sum(1 for e in events if e["type"] == "shot_made")
    shots = sum(1 for e in events if e["type"].startswith("shot_"))
    doc["source"] = (f"detected positions, {len(events)} events tagged by hand"
                     if events else doc["source"])
    Path(args.data).write_text(json.dumps(doc))
    print(f"\nwrote {args.data}: {len(events)} events, {made}/{shots} from the field")
    print("The positions in those events came from the tracking, not from clicking a")
    print("spot on a court diagram -- which is the only reason the shot chart means")
    print("anything.")


if __name__ == "__main__":
    main()
