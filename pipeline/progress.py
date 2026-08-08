"""Progress files for long pipeline runs, and a watcher that reads them.

The SAM2 run made the case: two and a half hours of GPU at 100% with nothing
visible, because Python buffers stdout on background processes and the script
wrote its output once, at the end. Progress belongs in files, not in a
terminal's buffer.

A script reports like this:

    from progress import Progress
    prog = Progress("detect-nba", total=6200)
    for ...:
        prog.step(note=f"frame {idx}")
    prog.done()

and anyone watches like this, in any terminal:

    python pipeline/progress.py            # one snapshot
    python pipeline/progress.py --watch    # refreshed every 2s

Files land in out/progress/<job>.json, written atomically (temp + replace) and
throttled to one write a second, so reporting costs nothing measurable. The
watcher flags a job whose file has gone quiet -- with a buffered terminal that
is invisible, with files it is the first thing you see.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "out" / "progress"


class Progress:
    def __init__(self, job, total=None, note=""):
        self.job = job
        self.total = total
        self.count = 0
        self.started = time.time()
        self._last_write = 0.0
        self.path = DIR / f"{job}.json"
        DIR.mkdir(parents=True, exist_ok=True)
        self._write("running", note, force=True)

    def step(self, n=1, note=""):
        self.count += n
        self._write("running", note)

    def set(self, count, note=""):
        self.count = count
        self._write("running", note)

    def done(self, note=""):
        self._write("done", note, force=True)

    def fail(self, note=""):
        self._write("failed", note, force=True)

    def _write(self, state, note, force=False):
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        self._last_write = now
        payload = {
            "job": self.job, "state": state,
            "done": self.count, "total": self.total,
            "note": note, "started": self.started, "updated": now,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)


def _fmt_eta(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return "-"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def snapshot():
    rows = []
    for f in sorted(DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        now = time.time()
        elapsed = now - d["started"]
        rate = d["done"] / elapsed if elapsed > 0 and d["done"] else 0
        eta = ((d["total"] - d["done"]) / rate) if (d.get("total") and rate) else None
        silent = now - d["updated"]
        state = d["state"]
        if state == "running" and silent > 90:
            state = f"quiet {_fmt_eta(silent)}"   # the wedged-or-dead flag
        if d.get("total"):
            frac = d["done"] / d["total"]
            bar = "#" * int(frac * 24)
            prog = f"[{bar:<24}] {frac * 100:3.0f}%  {d['done']}/{d['total']}"
        else:
            prog = f"{d['done']} done"
        rows.append(f"{d['job']:<18} {state:<10} {prog}  "
                    f"{rate:.1f}/s  eta {_fmt_eta(eta)}  {d.get('note', '')}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh every 2s")
    ap.add_argument("--clear", action="store_true", help="delete finished jobs' files")
    args = ap.parse_args()

    if args.clear:
        for f in DIR.glob("*.json"):
            try:
                if json.loads(f.read_text())["state"] in ("done", "failed"):
                    f.unlink()
                    print(f"cleared {f.stem}")
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        return

    while True:
        rows = snapshot()
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(time.strftime("%H:%M:%S"), "- out/progress/")
        if rows:
            print("\n".join(rows))
        else:
            print("  (no jobs have reported)")
        if not args.watch:
            break
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
