"""Correct a track's number by hand, and record that a human did it.

The OCR is good but not perfect: on the NYK @ DET segment it read Josh
Hart's 3 as a 9, which the roster then refused to name because no Knick
wears 9. The pipeline is right to refuse rather than guess, and a human
reading the jersey is the cheapest way to settle it.

Editing the JSON by hand would work once and be lost on the next run, and
would leave no trace of which identities a model produced and which a person
did. This applies corrections as a step, keeps them in the file, and marks
them:

    python pipeline/fix_identity.py --identities out/identities_det3006.json \
        --rosters web/data/rosters_det.json --set 2=3

The number is re-looked-up against the roster of the club the track was
already clustered into, so a correction cannot silently move a player
between teams. Pass --club to override that too.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", required=True)
    ap.add_argument("--rosters", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="TID=NUMBER",
                    help="repeatable; the number as it reads on the jersey")
    ap.add_argument("--club", help="force the club for every --set in this run")
    ap.add_argument("--out", help="defaults to updating --identities in place")
    args = ap.parse_args()

    path = Path(args.identities)
    data = json.loads(path.read_text(encoding="utf-8"))
    rosters = json.loads(Path(args.rosters).read_text(encoding="utf-8"))
    by_club = {c: {str(p["num"]): p["name"] for p in rosters[c]}
               for c in rosters if isinstance(rosters[c], list)}

    for item in args.set:
        tid, _, number = item.partition("=")
        tid, number = tid.strip(), number.strip()
        if tid not in data["identities"]:
            raise SystemExit(f"no track {tid} in {path}")
        rec = data["identities"][tid]
        club = args.club or rec.get("club")
        if club not in by_club:
            raise SystemExit(
                f"track {tid} has club {club!r}, which is not in the roster "
                f"({', '.join(by_club)}). Pass --club.")
        name = by_club[club].get(number)
        if name is None:
            raise SystemExit(
                f"{club} has no number {number}. Numbers on that roster: "
                f"{', '.join(sorted(by_club[club], key=lambda n: (len(n), n)))}")

        was = f"#{rec.get('number')} {rec.get('name') or '(unnamed)'}"
        rec["number"] = number
        rec["name"] = name
        rec["club"] = club
        rec["corrected"] = True      # a person read this jersey, not a model
        data["labels"][tid] = f"#{number} {name}"
        print(f"track {tid}: {was}  ->  #{number} {name} ({club})")

    out = Path(args.out or args.identities)
    out.write_text(json.dumps(data), encoding="utf-8")
    n = sum(1 for v in data["identities"].values() if v.get("corrected"))
    print(f"\nwrote {out} ({n} identities now marked corrected)")


if __name__ == "__main__":
    main()
