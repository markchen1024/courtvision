"""Progress files for long pipeline runs, and the tools that read them.

The SAM2 run made the case: two and a half hours of GPU at 100% with nothing
visible, because ultralytics buffered every result and the reporting loop never
ran. Progress belongs in files, not in a terminal's buffer.

A script reports like this:

    from progress import Progress
    prog = Progress("detect-nba", total=6200)
    for ...:
        prog.step(note=f"frame {idx}")
    prog.done()

and there are two ways to watch, both dependency-free:

    python pipeline/progress.py --serve    # http://localhost:8799 in a browser
    python pipeline/progress.py --watch    # terminal, refreshed every 2s

Files land in out/progress/<job>.json, written atomically (temp + replace) and
throttled to one write a second, so reporting costs nothing measurable. Both
viewers flag a job whose file has gone quiet -- with a buffered terminal that
is invisible, here it is the first thing you see.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "out" / "progress"
PORT = 8799


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
        # On Windows, os.replace is denied while a reader (the watcher polling)
        # has the target open. Reporting must never kill the job it reports on:
        # drop the update and move on. Final states get a few retries instead.
        for attempt in range(5 if force else 1):
            try:
                tmp.write_text(json.dumps(payload))
                os.replace(tmp, self.path)
                return
            except OSError:
                if force:
                    time.sleep(0.05)


def read_jobs():
    """Every job file, parsed, with rate/eta/silence computed the one place."""
    # totals.hint fills in a total the reporting script didn't know (or, as with
    # the first SAM2 run, predated): {"sam2": 750}. Not .json, so it isn't a job.
    try:
        hints = json.loads((DIR / "totals.hint").read_text())
    except (json.JSONDecodeError, OSError):
        hints = {}
    jobs = []
    for f in sorted(DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not d.get("total"):
            d["total"] = hints.get(d["job"])
        now = time.time()
        elapsed = now - d["started"]
        rate = d["done"] / elapsed if elapsed > 0 and d["done"] else 0
        d["rate"] = rate
        d["eta"] = ((d["total"] - d["done"]) / rate) if (d.get("total") and rate) else None
        d["elapsed"] = elapsed
        d["silent"] = now - d["updated"]
        jobs.append(d)
    return jobs


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
    for d in read_jobs():
        state = d["state"]
        if state == "running" and d["silent"] > 90:
            state = f"quiet {_fmt_eta(d['silent'])}"   # the wedged-or-dead flag
        if d.get("total"):
            frac = d["done"] / d["total"]
            bar = "#" * int(frac * 24)
            prog = f"[{bar:<24}] {frac * 100:3.0f}%  {d['done']}/{d['total']}"
        else:
            prog = f"{d['done']} done"
        rows.append(f"{d['job']:<18} {state:<10} {prog}  "
                    f"{d['rate']:.1f}/s  eta {_fmt_eta(d['eta'])}  {d.get('note', '')}")
    return rows


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>courtvision — pipeline progress</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 2rem; background: #10161a; color: #f5f8fa;
         font: 14px/1.5 ui-monospace, Consolas, monospace; }
  h1 { font-size: 1rem; font-weight: 600; color: #8a9ba8; margin: 0 0 1.25rem; }
  h1 small { font-weight: 400; }
  .stage { margin-bottom: 1.4rem; }
  .stage-head { display: flex; align-items: baseline; gap: .7rem;
                padding-bottom: .45rem; margin-bottom: .6rem;
                border-bottom: 1px solid #2f343c; }
  .stage-no { font-size: .72rem; color: #10161a; background: #8a9ba8;
              border-radius: 4px; padding: .1rem .45rem; font-weight: 700; }
  .stage-head.active .stage-no { background: #3dcc91; }
  .stage-name { font-weight: 700; letter-spacing: .02em; }
  .stage-sum { margin-left: auto; font-size: .78rem; color: #5f6b7c; }
  .stage-idle { color: #444f5a; font-size: .8rem; padding: .15rem 0 .1rem; }
  .job { background: #182026; border: 1px solid #2f343c; border-radius: 6px;
         padding: 1rem 1.25rem; margin-bottom: .75rem; }
  .top { display: flex; gap: .75rem; align-items: baseline; flex-wrap: wrap; }
  .name { font-size: 1.05rem; font-weight: 700; }
  .chip { font-size: .72rem; padding: .1rem .5rem; border-radius: 999px;
          text-transform: uppercase; letter-spacing: .05em; }
  .running { background: #0f4a37; color: #3dcc91; }
  .done    { background: #1d3a5c; color: #48aff0; }
  .failed  { background: #5c2020; color: #ff7373; }
  .quiet   { background: #5c4813; color: #ffc940; }
  .nums { margin-left: auto; color: #8a9ba8; }
  .track { height: 10px; background: #0d1216; border-radius: 999px;
           margin: .65rem 0 .35rem; overflow: hidden; }
  .fill { height: 100%; background: #3dcc91; border-radius: 999px;
          transition: width .8s ease; }
  .fill.indet { width: 30% !important; opacity: .55;
                animation: slide 1.6s ease-in-out infinite alternate; }
  @keyframes slide { from { margin-left: 0 } to { margin-left: 70% } }
  .note { color: #8a9ba8; font-size: .82rem; }
  .empty { color: #5f6b7c; padding: 3rem 0; text-align: center; }
</style>
<h1>pipeline progress <small id="clock"></small></h1>
<div id="jobs"><div class="empty">(no jobs have reported)</div></div>
<script>
// The pipeline's shape, in run order. Job files map onto these stages;
// anything unrecognised lands in "other" rather than vanishing.
const STAGES = [
  ["Calibration",          ["keypoints"]],
  ["Detection & tracking", ["detect", "dense-detect", "sam2", "sam3"]],
  ["Identity",             ["identify", "shirts", "resnet-ocr"]],
  ["Events",               ["shot-events"]],
  ["Render",               ["render", "final-render"]],
];

const eta = s => s == null || s < 0 ? "-"
  : s < 90 ? Math.round(s) + "s"
  : s < 5400 ? Math.round(s / 60) + "m"
  : (s / 3600).toFixed(1) + "h";

function jobCard(d) {
  const quiet = d.state === "running" && d.silent > 90;
  const cls = quiet ? "quiet" : d.state;
  const label = quiet ? "quiet " + eta(d.silent) : d.state;
  const pct = d.total ? d.done / d.total * 100 : null;
  const nums = d.total
    ? `${pct.toFixed(0)}% · ${d.done}/${d.total} · ${d.rate.toFixed(1)}/s · eta ${eta(d.eta)}`
    : `${d.done} done · ${d.rate.toFixed(1)}/s · running ${eta(d.elapsed)}`;
  const fill = d.state === "failed" ? "#ff7373" : quiet ? "#ffc940"
             : d.state === "done" ? "#48aff0" : "#3dcc91";
  return `<div class="job">
    <div class="top"><span class="name">${d.job}</span>
      <span class="chip ${cls}">${label}</span>
      <span class="nums">${nums}</span></div>
    <div class="track"><div class="fill${pct == null && d.state === "running" ? " indet" : ""}"
      style="width:${pct == null ? 100 : Math.max(pct, 1.5)}%;background:${fill}"></div></div>
    <div class="note">${d.note || ""}</div>
  </div>`;
}

function stageSection(no, name, jobs) {
  const running = jobs.filter(j => j.state === "running").length;
  const done = jobs.filter(j => j.state === "done").length;
  const sum = jobs.length
    ? [running && running + " running", done && done + " done",
       (jobs.length - running - done) && (jobs.length - running - done) + " failed"]
        .filter(Boolean).join(" · ")
    : "";
  const body = jobs.length ? jobs.map(jobCard).join("")
                           : '<div class="stage-idle">idle</div>';
  return `<div class="stage">
    <div class="stage-head${running ? " active" : ""}">
      <span class="stage-no">${no}</span>
      <span class="stage-name">${name}</span>
      <span class="stage-sum">${sum}</span></div>
    ${body}</div>`;
}

async function tick() {
  try {
    const jobs = await (await fetch("/jobs", {cache: "no-store"})).json();
    document.getElementById("clock").textContent =
      "· " + new Date().toLocaleTimeString();
    const claimed = new Set();
    const sections = STAGES.map(([name, names], i) => {
      const mine = jobs.filter(j => names.includes(j.job));
      mine.forEach(j => claimed.add(j.job));
      return stageSection(i + 1, name, mine);
    });
    const rest = jobs.filter(j => !claimed.has(j.job));
    if (rest.length) sections.push(stageSection("·", "Other", rest));
    document.getElementById("jobs").innerHTML = sections.join("");
  } catch (e) { /* server restarting; keep polling */ }
}
tick();
setInterval(tick, 1500);
</script>
"""


def serve(port):
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/jobs":
                body = json.dumps(read_jobs()).encode()
                ctype = "application/json"
            else:
                body = PAGE.encode("utf-8")
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    DIR.mkdir(parents=True, exist_ok=True)
    print(f"progress page: http://localhost:{port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true",
                    help=f"serve a live progress page on http://localhost:{PORT}")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--watch", action="store_true", help="refresh every 2s")
    ap.add_argument("--clear", action="store_true", help="delete finished jobs' files")
    args = ap.parse_args()

    if args.serve:
        serve(args.port)
        return

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
