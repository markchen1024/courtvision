"""Generate synthetic tracking data so the viewer can be built before the CV works.

The real pipeline will emit exactly this shape, so the front end never has to change.
Court coordinates are in metres on a FIBA court (28 x 15), origin at the bottom-left
corner as seen in the top-down view.

    python pipeline/make_sample_data.py --seconds 90 --out web/data/sample.json
"""

import argparse
import json
import math
import random
from pathlib import Path

COURT_L = 28.0  # length, metres
COURT_W = 15.0  # width, metres
HZ = 5  # position samples per second

HOME_NUMBERS = [4, 7, 11, 21, 23]
AWAY_NUMBERS = [3, 8, 12, 15, 30]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_players():
    players = []
    for n in HOME_NUMBERS:
        players.append({"id": len(players) + 1, "team": "home", "number": n})
    for n in AWAY_NUMBERS:
        players.append({"id": len(players) + 1, "team": "away", "number": n})
    return players


def build_tracks(players, seconds, rng):
    """Smooth random walk, with both teams drifting toward whichever end is in play."""
    n_frames = int(seconds * HZ)

    state = {}
    for p in players:
        x = rng.uniform(4, COURT_L - 4)
        y = rng.uniform(2, COURT_W - 2)
        state[p["id"]] = [x, y, 0.0, 0.0]  # x, y, vx, vy

    frames = []
    for i in range(n_frames):
        t = round(i / HZ, 2)
        # possessions flip roughly every 14 seconds
        attacking_left = (int(t) // 14) % 2 == 0
        target_x = 6.0 if attacking_left else COURT_L - 6.0

        positions = []
        for p in players:
            x, y, vx, vy = state[p["id"]]
            # pull toward the live end of the floor, plus jitter
            vx += (target_x - x) * 0.012 + rng.gauss(0, 0.09)
            vy += (COURT_W / 2 - y) * 0.010 + rng.gauss(0, 0.11)
            vx, vy = clamp(vx, -0.7, 0.7), clamp(vy, -0.7, 0.7)
            x = clamp(x + vx, 0.8, COURT_L - 0.8)
            y = clamp(y + vy, 0.8, COURT_W - 0.8)
            state[p["id"]] = [x, y, vx * 0.86, vy * 0.86]
            positions.append({"id": p["id"], "x": round(x, 2), "y": round(y, 2)})

        frames.append({"t": t, "positions": positions})

    return frames


def nearest_position(frames, t, player_id):
    idx = min(int(t * HZ), len(frames) - 1)
    for pos in frames[idx]["positions"]:
        if pos["id"] == player_id:
            return pos["x"], pos["y"]
    return COURT_L / 2, COURT_W / 2


def build_events(players, frames, seconds, rng):
    """Shots, rebounds, assists and turnovers at plausible intervals."""
    events = []
    home = [p for p in players if p["team"] == "home"]
    away = [p for p in players if p["team"] == "away"]

    t = 4.0
    while t < seconds - 2:
        attacking_left = (int(t) // 14) % 2 == 0
        offence = home if attacking_left else away
        defence = away if attacking_left else home
        shooter = rng.choice(offence)
        x, y = nearest_position(frames, t, shooter["id"])

        basket_x = 1.6 if attacking_left else COURT_L - 1.6
        dist = math.hypot(x - basket_x, y - COURT_W / 2)
        three = dist > 6.75
        points = 3 if three else 2
        made = rng.random() < (0.34 if three else 0.52)

        if made and rng.random() < 0.55:
            passer = rng.choice([p for p in offence if p["id"] != shooter["id"]])
            events.append(
                {"t": round(t - 0.8, 2), "type": "assist", "player": passer["id"]}
            )

        events.append(
            {
                "t": round(t, 2),
                "type": "shot_made" if made else "shot_missed",
                "player": shooter["id"],
                "team": shooter["team"],
                "x": round(x, 2),
                "y": round(y, 2),
                "points": points,
            }
        )

        if not made:
            rebounder = rng.choice(offence + defence)
            events.append(
                {
                    "t": round(t + 1.1, 2),
                    "type": "rebound",
                    "player": rebounder["id"],
                    "team": rebounder["team"],
                }
            )

        if rng.random() < 0.18:
            loser = rng.choice(offence)
            events.append(
                {
                    "t": round(t + 2.4, 2),
                    "type": "turnover",
                    "player": loser["id"],
                    "team": loser["team"],
                }
            )

        t += rng.uniform(5.0, 9.0)

    events.sort(key=lambda e: e["t"])
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="web/data/sample.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    players = build_players()
    frames = build_tracks(players, args.seconds, rng)
    events = build_events(players, frames, args.seconds, rng)

    doc = {
        "source": "synthetic",
        "video": {"duration": args.seconds, "hz": HZ},
        "court": {"length_m": COURT_L, "width_m": COURT_W},
        "players": players,
        "frames": frames,
        "events": events,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")

    made = sum(1 for e in events if e["type"] == "shot_made")
    shots = sum(1 for e in events if e["type"].startswith("shot_"))
    print(f"wrote {out}  ({len(frames)} frames, {len(events)} events, {made}/{shots} FG)")


if __name__ == "__main__":
    main()
