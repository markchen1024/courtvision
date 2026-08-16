"""Fetch the official play-by-play for the demo game, slimmed for the viewer.

The Timeline tab showed invented rows for a year because event detection is
not implemented -- but the events themselves are public record. ESPN's game
summary carries all 461 plays of NYK @ DET game 4 with period, clock, text,
running score and scoring flags: the same provenance as the box score the
page already uses, and the panel says so.

The one thing ESPN cannot know is which plays are on film. The homepage clip
covers Q3 2:04 to 1:21 (read off the broadcast scoreboard: 1:24 shows at
t=40s), so plays inside that window get a clip timestamp and the viewer
highlights them.

    python pipeline/fetch_pbp.py        # writes web/data/pbp.json
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAME_ID = "401768043"   # NYK @ DET, 2025 East first round game 4
URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
       f"?event={GAME_ID}")

# The homepage clip in game-clock terms: Q3, from FILM_T0 seconds remaining,
# running FILM_LEN seconds of continuous action.
FILM_PERIOD = 3
FILM_T0 = 124.0        # 2:04 on the clock at clip t=0
FILM_LEN = 42.6


def clock_seconds(display):
    if ":" in display:
        m, s = display.split(":")
        return int(m) * 60 + float(s)
    return float(display)


def kind_of(play):
    text = play.get("text", "").lower()
    type_text = play.get("type", {}).get("text", "").lower()
    if "rebound" in text or "rebound" in type_text:
        return "board"
    if "turnover" in type_text or "turnover" in text or "steal" in text:
        return "loss"
    if play.get("shootingPlay") or "makes" in text or "misses" in text:
        return "score"
    return "other"


def main():
    with urllib.request.urlopen(URL, timeout=30) as r:
        doc = json.load(r)

    # team id -> abbreviation, from the boxscore half of the same document
    abbrev = {}
    for side in doc.get("boxscore", {}).get("teams", []):
        team = side.get("team", {})
        abbrev[team.get("id")] = team.get("abbreviation")

    plays = []
    for p in doc.get("plays", []):
        period = p.get("period", {}).get("number")
        clock = p.get("clock", {}).get("displayValue", "")
        if period is None or not clock:
            continue
        row = {
            "q": period,
            "clock": clock,
            "text": p.get("text", ""),
            "away": p.get("awayScore"),
            "home": p.get("homeScore"),
            "kind": kind_of(p),
            "team": abbrev.get(p.get("team", {}).get("id")),
        }
        if period == FILM_PERIOD:
            t = FILM_T0 - clock_seconds(clock)
            if 0 <= t <= FILM_LEN:
                row["film_t"] = round(t, 1)
        plays.append(row)

    out = ROOT / "web" / "data" / "pbp.json"
    out.write_text(json.dumps({
        "source": f"ESPN play-by-play, gameId {GAME_ID}",
        "game": "NYK @ DET · 2025 East first round, game 4",
        "film_window": {"period": FILM_PERIOD, "from": "2:04", "to": "1:21"},
        "plays": plays,
    }), encoding="utf-8")

    on_film = [p for p in plays if "film_t" in p]
    kinds = {}
    for p in plays:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"wrote {out.relative_to(ROOT)}: {len(plays)} plays, kinds {kinds}")
    print("on film:")
    for p in on_film:
        print(f"  Q{p['q']} {p['clock']} (clip {p['film_t']}s) {p['text']}")


if __name__ == "__main__":
    main()
