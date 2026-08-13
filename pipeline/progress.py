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

A finished job also leaves a copy in out/progress/history/, keyed by its start
time. The live files answer "what is running"; on any day nothing is running
that is an empty page, and the interesting question is the other one -- what
this clip went through to become the render. History is grouped by source
video for exactly that: one row per segment, its stages underneath, the
artifact each stage produced, and the provenance (weights, model ids,
thresholds, commit) that the run actually used rather than what a docstring
claims it used.
"""

import argparse
import atexit
import json
import os
import sys
import time
import traceback as _traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "out" / "progress"
HISTORY = DIR / "history"
PORT = 8799

_COMMIT = None


def _git_commit():
    """The working tree's commit, so an archived run says which code made it.

    Cached: this is asked once per Progress and a subprocess per job would be
    silly. Returns None outside a checkout rather than raising -- provenance
    that fails is a missing field, never a dead run.
    """
    global _COMMIT
    if _COMMIT is None:
        import subprocess
        try:
            r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=5)
            _COMMIT = (r.stdout or "").strip() or ""
        except (OSError, subprocess.SubprocessError):
            _COMMIT = ""
    return _COMMIT or None


# Every Progress that has been constructed and not yet finished, innermost
# last. Scripts like find_segments.py and run_tutorial.py build several in
# sequence, so "the run that is happening now" is the top of this stack, not
# some single global.
_LIVE = []


class Progress:
    def __init__(self, job, total=None, note="", video=None, folder=None,
                 artifact=None, meta=None):
        self.job = job
        self.total = total
        self.count = 0
        self.started = time.time()
        self._last_write = 0.0
        self._final = None
        # What this job is chewing through, and where its output lands. Both
        # are for the watcher: a still from the clip says more about whether a
        # run is on the right footage than any counter, and the folder button
        # saves hunting for out/ in a file manager.
        self.video = str(video) if video else None
        self.folder = str(folder) if folder else "out"
        # The one file this run exists to produce, and what it was configured
        # with. Both are for the history view rather than the live one: a
        # finished stage is worth nothing on screen unless you can play what it
        # made and see which weights made it.
        self.artifact = str(artifact) if artifact else None
        self.meta = dict(meta) if meta else {}
        self.commit = _git_commit()
        self.traceback = None
        self.path = DIR / f"{job}.json"
        DIR.mkdir(parents=True, exist_ok=True)
        _LIVE.append(self)
        self._write("running", note, force=True)

    def step(self, n=1, note=""):
        self.count += n
        self._write("running", note)

    def set(self, count, note=""):
        self.count = count
        self._write("running", note)

    def info(self, **fields):
        """Record provenance discovered after construction.

        Half of what is worth knowing -- the checkpoint a loader actually
        resolved, the thresholds after argparse defaults land -- is only known
        once the job is already running, and making every script assemble it
        before its first frame would be the kind of bookkeeping people quietly
        stop doing.
        """
        self.meta.update({k: v for k, v in fields.items() if v is not None})

    def done(self, note=""):
        self._finish("done", note)

    def fail(self, note="", traceback=None):
        self.traceback = traceback or self.traceback
        self._finish("failed", note)

    def _finish(self, state, note):
        if self._final:                     # done() then an atexit sweep, say
            return
        self._final = state
        if self in _LIVE:
            _LIVE.remove(self)
        self._write(state, note, force=True)
        self._archive(state, note)

    def _payload(self, state, note, now):
        return {
            "job": self.job, "state": state,
            "done": self.count, "total": self.total,
            "note": note, "started": self.started, "updated": now,
            "video": self.video, "folder": self.folder,
            "artifact": self.artifact, "meta": self.meta,
            "commit": self.commit, "traceback": self.traceback,
        }

    def _archive(self, state, note):
        """Keep the finished run. Start time is already a unique id per run."""
        try:
            HISTORY.mkdir(parents=True, exist_ok=True)
            path = HISTORY / f"{self.job}-{int(self.started)}.json"
            path.write_text(json.dumps(self._payload(state, note, time.time())))
        except OSError:
            pass                            # never let bookkeeping kill a run

    def _write(self, state, note, force=False):
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        self._last_write = now
        payload = self._payload(state, note, now)
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


# --- Finishing a run that did not finish itself ------------------------------
#
# Nothing in pipeline/ has ever called fail(). A crashed script simply stopped
# writing, so the page showed it as "quiet", then "stale", forever -- and the
# archive would have filled with runs frozen mid-count. Two hooks close that.

def _on_exception(exc_type, exc, tb):
    text = "".join(_traceback.format_exception(exc_type, exc, tb))
    note = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else \
        f"{exc_type.__name__}: {exc}"[:200]
    for prog in list(_LIVE):
        prog.fail(note=note, traceback=text)
    _PREV_HOOK(exc_type, exc, tb)           # the terminal still goes red


def _on_exit():
    """A clean interpreter exit with a job still open means the script simply
    never called done(). track_bytetrack.py has never called it -- dense-detect
    succeeded and sat on the page as a corpse. That is a finished run, not a
    failed one; say so, and do not invent a failure the run did not have."""
    for prog in list(_LIVE):
        prog.done(note="ended without done()")


_PREV_HOOK = sys.excepthook
sys.excepthook = _on_exception
atexit.register(_on_exit)


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


def read_history(limit=400):
    """Archived runs, newest first, with the fields the history view derives."""
    runs = []
    for f in sorted(HISTORY.glob("*.json"), reverse=True)[:limit]:
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        d["run"] = f.stem
        d["elapsed"] = max(0.0, d.get("updated", 0) - d.get("started", 0))
        d["rate"] = d["done"] / d["elapsed"] if d["elapsed"] and d.get("done") else 0
        runs.append(d)
    runs.sort(key=lambda d: d.get("started", 0), reverse=True)
    return runs


def seed_history():
    """Archive finished live jobs that predate the archive.

    Twelve real runs were sitting in out/progress/ when history was added,
    including the whole seg_01m10.87s_19s chain. Throwing them away to start
    the archive clean would have been the wrong trade.
    """
    HISTORY.mkdir(parents=True, exist_ok=True)
    seeded = 0
    for f in sorted(DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("state") not in ("done", "failed") or not d.get("started"):
            continue
        target = HISTORY / f"{d['job']}-{int(d['started'])}.json"
        if target.exists():
            continue
        d.setdefault("artifact", None)
        d.setdefault("meta", {})
        d.setdefault("commit", None)
        try:
            target.write_text(json.dumps(d))
            seeded += 1
        except OSError:
            pass
    return seeded


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
<title>courtvision — pipeline</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500&display=swap');
  :root {
    --void:#08090c; --surface:#0e1015; --raised:#151920; --inset:#050608;
    --line:#23272f; --line-soft:#161a20; --line-strong:#333a45;
    --fg:#eef1f5; --muted:#99a2af; --subtle:#6a7383; --inverse:#08090c;
    --signal:#c8f031; --signal-line:#3e4d12; --teal:#3bc9a8;
    --amber:#ffc940; --red:#ff7373;
    --font-display:'Space Grotesk',ui-sans-serif,system-ui,sans-serif;
    --font-sans:'Inter',ui-sans-serif,system-ui,sans-serif;
    --font-mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--void); color:var(--fg);
         font:400 14px/1.5 var(--font-sans); }
  .shell { max-width: 880px; margin: 0 auto; padding: 2.2rem 1.5rem 4rem; }

  .topbar { display:flex; align-items:center; gap:.9rem; margin-bottom:2rem; }
  .brand { font:700 18px/1 var(--font-display); letter-spacing:-.01em; }
  .brand i { display:inline-block; width:9px; height:9px; border-radius:999px;
             background:var(--signal); margin-right:.55rem; }
  .toplabel { font:500 11px/1 var(--font-mono); letter-spacing:.14em;
              color:var(--subtle); text-transform:uppercase; padding-top:2px; }
  .clock { margin-left:auto; font:400 11.5px/1 var(--font-mono); color:var(--subtle); }
  .ghost { font:500 11px/1 var(--font-mono); letter-spacing:.06em; color:var(--muted);
           background:none; border:1px solid var(--line); border-radius:999px;
           padding:.45rem .8rem; cursor:pointer; }
  .ghost:hover { color:var(--fg); border-color:var(--line-strong); }

  .stage { display:grid; grid-template-columns: 34px 1fr; gap: 0 1.1rem; }
  .rail { display:flex; flex-direction:column; align-items:center; }
  .node { width:34px; height:34px; border-radius:999px; flex:none;
          display:grid; place-items:center; font:700 13px/1 var(--font-mono);
          background:var(--raised); border:1px solid var(--line);
          color:var(--subtle); }
  .stage.active .node { background:var(--signal); border-color:var(--signal);
                        color:var(--inverse); }
  .stage.done-all .node { color:var(--teal); border-color:var(--teal); }
  .railline { width:1px; flex:1; background:var(--line-soft); margin:6px 0; }

  .stagebody { padding-bottom: 1.7rem; min-width:0; }
  .stagehead { display:flex; align-items:baseline; gap:.8rem; }
  .stagename { font:700 15px/1.2 var(--font-display); }
  .stagesum { margin-left:auto; font:400 11px/1 var(--font-mono); color:var(--subtle);
              white-space:nowrap; }
  .stagedesc { font:400 12.5px/1.55 var(--font-sans); color:var(--subtle);
               margin:.25rem 0 .75rem; }
  .idle { font:400 11.5px/1 var(--font-mono); color:var(--line-strong);
          border:1px dashed var(--line-soft); border-radius:.5rem;
          padding:.7rem .9rem; }

  .job { position:relative; background:var(--surface); border:1px solid var(--line-soft);
         border-radius:.75rem; padding:.9rem 1.1rem .8rem; margin-bottom:.6rem; }
  .job.dim { opacity:.55; }
  .jobtop { display:flex; align-items:center; gap:.7rem; }
  .jobname { font:700 13px/1 var(--font-mono); }
  .chip { font:500 10px/1 var(--font-mono); letter-spacing:.08em;
          text-transform:uppercase; border-radius:999px; padding:.3rem .55rem; }
  .chip.running { background:var(--signal-line); color:var(--signal); }
  .chip.done    { background:#123830; color:var(--teal); }
  .chip.failed  { background:#3d1b1b; color:var(--red); }
  .chip.quiet, .chip.stale { background:#3d3312; color:var(--amber); }
  .nums { margin-left:auto; font:400 11px/1 var(--font-mono); color:var(--subtle);
          white-space:nowrap; }
  .dismiss { flex:none; width:22px; height:22px; border-radius:999px; border:none;
             background:none; color:var(--subtle); font:400 13px/1 var(--font-mono);
             cursor:pointer; margin-left:.1rem; }
  .dismiss:hover { color:var(--fg); background:var(--raised); }
  .track { height:7px; background:var(--inset); border-radius:999px;
           margin:.65rem 0 .4rem; overflow:hidden; }
  .fill { height:100%; border-radius:999px; transition:width .8s ease; }
  .fill.indet { width:30% !important; opacity:.5;
                animation:slide 1.6s ease-in-out infinite alternate; }
  @keyframes slide { from { margin-left:0 } to { margin-left:70% } }
  .note { font:400 11px/1.4 var(--font-mono); color:var(--subtle); }
  .empty { color:var(--subtle); }
  .body { display:flex; gap:.9rem; align-items:flex-start; }
  .bars { flex:1; min-width:0; }
  .shot { width:168px; flex:none; aspect-ratio:16/9; object-fit:cover;
          border-radius:.45rem; border:1px solid var(--line);
          background:var(--inset); margin-top:.55rem; }
  .src { font:400 10.5px/1.4 var(--font-mono); color:var(--line-strong);
         margin-top:.3rem; overflow:hidden; text-overflow:ellipsis;
         white-space:nowrap; }

  .tabs { display:flex; gap:.35rem; margin:0 0 1.6rem; }
  .tab { font:500 11px/1 var(--font-mono); letter-spacing:.1em;
         text-transform:uppercase; color:var(--subtle); background:none;
         border:1px solid transparent; border-radius:999px; padding:.5rem .95rem;
         cursor:pointer; }
  .tab:hover { color:var(--fg); }
  .tab.on { color:var(--inverse); background:var(--signal); }
  .tab b { font:500 11px/1 var(--font-mono); opacity:.65; margin-left:.4rem; }

  /* History: one row per source clip, its stages folded underneath. */
  .clip { background:var(--surface); border:1px solid var(--line-soft);
          border-radius:.75rem; margin-bottom:.7rem; overflow:hidden; }
  .cliphead { display:flex; align-items:center; gap:.9rem; padding:.85rem 1.1rem;
              cursor:pointer; }
  .cliphead:hover { background:var(--raised); }
  .clipshot { width:88px; flex:none; aspect-ratio:16/9; object-fit:cover;
              border-radius:.35rem; border:1px solid var(--line);
              background:var(--inset); }
  .clipname { font:700 13px/1.3 var(--font-mono); min-width:0;
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .clipsum { font:400 11px/1.4 var(--font-mono); color:var(--subtle);
             margin-top:.25rem; }
  .caret { margin-left:auto; flex:none; color:var(--subtle);
           font:400 11px/1 var(--font-mono); }
  .clipstages { border-top:1px solid var(--line-soft); padding:.3rem 1.1rem 1rem; }

  .srow { display:grid; grid-template-columns:1fr auto; gap:.15rem .8rem;
          padding:.75rem 0; border-bottom:1px solid var(--line-soft); }
  .srow:last-child { border-bottom:none; }
  .srowtop { display:flex; align-items:center; gap:.6rem; min-width:0; }
  .srowname { font:700 12px/1 var(--font-mono); }
  .srowtime { font:400 11px/1 var(--font-mono); color:var(--subtle);
              white-space:nowrap; text-align:right; }
  .srownote { grid-column:1/-1; font:400 11px/1.5 var(--font-mono);
              color:var(--subtle); }
  .kvs { grid-column:1/-1; display:flex; flex-wrap:wrap; gap:.3rem;
         margin-top:.45rem; }
  .kv { font:400 10px/1 var(--font-mono); color:var(--muted);
        background:var(--inset); border:1px solid var(--line-soft);
        border-radius:.3rem; padding:.32rem .5rem; }
  .kv b { color:var(--subtle); font-weight:400; }
  .kv.commit { color:var(--teal); }
  .art { grid-column:1/-1; width:100%; max-width:460px; margin-top:.6rem;
         border-radius:.45rem; border:1px solid var(--line);
         background:var(--inset); display:block; }
  .artlab { grid-column:1/-1; font:400 10.5px/1.4 var(--font-mono);
            color:var(--line-strong); margin-top:.3rem; }
  .trace { grid-column:1/-1; margin-top:.5rem; padding:.7rem .8rem;
           background:var(--inset); border:1px solid #3d1b1b; border-radius:.45rem;
           font:400 10.5px/1.5 var(--font-mono); color:var(--red);
           white-space:pre-wrap; max-height:220px; overflow:auto; }
  @media (max-width: 620px) { .shot, .clipshot { display:none; } }
</style>
<div class="shell">
  <div class="topbar">
    <span class="brand"><i></i>courtvision</span>
    <span class="toplabel">Pipeline monitor</span>
    <span class="clock" id="clock"></span>
    <button class="ghost" onclick="openOut()">Open out/</button>
    <button class="ghost" onclick="clearFinished()">Clear finished</button>
  </div>
  <div class="tabs">
    <button class="tab" id="tab-live" onclick="setView('live')">Live<b id="n-live"></b></button>
    <button class="tab" id="tab-history" onclick="setView('history')">History<b id="n-history"></b></button>
  </div>
  <div id="jobs"><div class="empty">(no jobs have reported)</div></div>
  <div id="history" hidden></div>
</div>
<script>
// The pipeline's shape, in run order, with one honest sentence each.
//
// Names are matched as prefixes, not equality. The list used to hold exact
// names and had gone stale against the scripts: sam2-tutorial, check-lineup,
// find-segments-cuts, find-segments-lineup, detect-cuts, review-detection,
// review-build, project-tutorial, legibility, harvest and tutorial-crops all
// piled into Other, which was most of what actually runs. A prefix also means
// the next sam2-something lands in the right stage without an edit here.
const STAGES = [
  ["Scouting", "Shot cuts found, then frames scored on whether a full lineup is visible — a segment worth an hour of GPU.",
   ["detect-cuts", "find-segments", "check-lineup"]],
  ["Detection & tracking", "Every player found on the prompt frame, then carried through the clip as identity tracks.",
   ["detect", "dense-detect", "sam2", "sam3"]],
  ["Identity", "Jersey numbers read and voted per track, teams clustered, roster joined — tracks become names.",
   ["identify", "shirts", "legibility"]],
  ["Court space", "Landmarks solved into a homography, players projected onto the court plan — pixels become metres.",
   ["keypoints", "project"]],
  ["Events", "Shot attempts from pose and rim signals. Outcomes stay hand-tagged, and the UI says so.",
   ["shot-events"]],
  ["Render & review", "Masks, name chips and the top-down court burned into video you can actually eyeball.",
   ["render", "final-render", "review"]],
  ["Labelling & training", "Crops harvested for Roboflow, and models fine-tuned on what comes back.",
   ["harvest", "resnet-ocr", "tutorial-crops", "train"]],
];
// longest prefix wins, so "detect-cuts" beats "detect" and lands in Scouting
const stageOf = job => {
  let best = -1, bestLen = 0;
  STAGES.forEach(([, , prefixes], i) => prefixes.forEach(p => {
    if ((job === p || job.startsWith(p)) && p.length > bestLen) {
      best = i; bestLen = p.length;
    }
  }));
  return best;
};
const STALE_S = 1800;   // quiet this long reads as a corpse, not a slow job

const eta = s => s == null || s < 0 ? "-"
  : s < 90 ? Math.round(s) + "s"
  : s < 5400 ? Math.round(s / 60) + "m"
  : (s / 3600).toFixed(1) + "h";

function jobCard(d) {
  const stale = d.state === "running" && d.silent > STALE_S;
  const quiet = !stale && d.state === "running" && d.silent > 90;
  const cls = stale ? "stale" : quiet ? "quiet" : d.state;
  const label = stale ? "stale " + eta(d.silent) : quiet ? "quiet " + eta(d.silent) : d.state;
  const pct = d.total ? d.done / d.total * 100 : null;
  const nums = d.total
    ? `${pct.toFixed(0)}% · ${d.done}/${d.total} · ${d.rate.toFixed(1)}/s · eta ${stale ? "-" : eta(d.eta)}`
    : `${d.done} done · ${d.rate.toFixed(1)}/s · ${eta(d.elapsed)}`;
  const fill = d.state === "failed" ? "var(--red)" : (stale || quiet) ? "var(--amber)"
             : d.state === "done" ? "var(--teal)" : "var(--signal)";
  const closable = stale || d.state !== "running";
  // A still from where the run has got to. Bucketed so it refreshes about
  // twenty times over a job instead of on every 1.5s poll -- enough to see the
  // clip advance, cheap enough that ffmpeg is not called in a loop.
  const bucket = pct == null ? 0 : Math.floor(pct / 5);
  const shot = d.video
    ? `<img class="shot" loading="lazy" alt=""
         src="/thumb?job=${encodeURIComponent(d.job)}&b=${bucket}"
         onerror="this.remove()">`
    : "";
  return `<div class="job${stale ? " dim" : ""}">
    <div class="jobtop"><span class="jobname">${d.job}</span>
      <span class="chip ${cls}">${label}</span>
      <span class="nums">${nums}</span>
      ${d.folder ? `<button class="dismiss" title="open ${d.folder}"
        onclick="openFolder('${d.job}')">&#128193;</button>` : ""}
      ${closable ? `<button class="dismiss" title="remove this job file" onclick="dismiss('${d.job}')">&times;</button>` : ""}</div>
    <div class="body">
      ${shot}
      <div class="bars">
        <div class="track"><div class="fill${pct == null && d.state === "running" && !stale ? " indet" : ""}"
          style="width:${pct == null ? 100 : Math.max(pct, 1.5)}%;background:${fill}"></div></div>
        <div class="note">${d.note || ""}</div>
        ${d.video ? `<div class="src" title="${d.video}">${d.video.split(/[\\\\/]/).pop()}</div>` : ""}
      </div>
    </div>
  </div>`;
}

function stageSection(no, name, desc, jobs, last) {
  const staleN = jobs.filter(j => j.state === "running" && j.silent > STALE_S).length;
  const running = jobs.filter(j => j.state === "running").length - staleN;
  const done = jobs.filter(j => j.state === "done").length;
  const failed = jobs.length - running - done - staleN;
  const parts = [];
  if (running) parts.push(running + " running");
  if (done) parts.push(done + " done");
  if (failed > 0) parts.push(failed + " failed");
  if (staleN) parts.push(staleN + " stale");
  const cls = running ? " active" : (jobs.length && done === jobs.length ? " done-all" : "");
  return `<div class="stage${cls}">
    <div class="rail"><div class="node">${no}</div>${last ? "" : '<div class="railline"></div>'}</div>
    <div class="stagebody">
      <div class="stagehead"><span class="stagename">${name}</span>
        <span class="stagesum">${parts.join(" · ")}</span></div>
      <div class="stagedesc">${desc}</div>
      ${jobs.length ? jobs.map(jobCard).join("") : '<div class="idle">idle</div>'}
    </div></div>`;
}

async function dismiss(job) {
  await fetch("/dismiss?job=" + encodeURIComponent(job), { method: "POST" });
  tick();
}
async function clearFinished() {
  await fetch("/clear-finished", { method: "POST" });
  tick();
}
async function openFolder(job) {
  await fetch("/open?job=" + encodeURIComponent(job), { method: "POST" });
}
async function openOut() {
  await fetch("/open", { method: "POST" });
}

// --- History -----------------------------------------------------------------
//
// A run archive listed as a flat stream of tasks is a log file with rounded
// corners. The unit worth showing is a clip: seg_01m10.87s_19s went through
// sam3, oncourt-filter, identify and final-render, and that chain is the thing
// to walk someone through. Group by source video, fold the stages underneath.

const clipName = p => p ? p.split(/[\\\\/]/).pop() : null;
const OPEN = new Set();          // which clips are expanded, across re-renders
let historySig = "";             // re-render only on change, so a playing
                                 // <video> is not torn out every poll

const when = ts => new Date(ts * 1000).toLocaleDateString(undefined,
  { month: "short", day: "numeric" });

function groupRuns(runs) {
  const groups = new Map();
  for (const r of runs) {
    const key = clipName(r.video) || "\\u0000not clip-scoped";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  // stages in the order they ran; clips with the most recent activity first
  const out = [...groups.entries()].map(([key, rs]) => {
    rs.sort((a, b) => a.started - b.started);
    return { key, runs: rs, latest: Math.max(...rs.map(r => r.started)) };
  });
  out.sort((a, b) => b.latest - a.latest);
  return out;
}

function metaChips(r) {
  const kvs = Object.entries(r.meta || {}).map(([k, v]) =>
    `<span class="kv"><b>${k}</b> ${v}</span>`);
  if (r.commit) kvs.push(`<span class="kv commit"><b>commit</b> ${r.commit}</span>`);
  return kvs.length ? `<div class="kvs">${kvs.join("")}</div>` : "";
}

function runRow(r) {
  const cls = r.state === "failed" ? "failed" : "done";
  const counts = r.total ? `${r.done}/${r.total}` : `${r.done}`;
  // Only what this stage produced. Falling back to the source clip put four
  // identical players in one clip's fold -- the source belongs to the clip, and
  // is shown once, at the top of the fold.
  const player = /\\.(mp4|webm)$/i.test(r.artifact || "") ? `
    <video class="art" controls preload="none"
      poster="/thumb?run=${encodeURIComponent(r.run)}"
      src="/file?run=${encodeURIComponent(r.run)}&what=artifact"></video>
    <div class="artlab">produced · ${clipName(r.artifact)}</div>` : "";
  return `<div class="srow">
    <div class="srowtop"><span class="srowname">${r.job}</span>
      <span class="chip ${cls}">${r.state}</span></div>
    <div class="srowtime">${eta(r.elapsed)} · ${counts} · ${when(r.started)}</div>
    ${r.note ? `<div class="srownote">${r.note}</div>` : ""}
    ${metaChips(r)}
    ${r.traceback ? `<div class="trace">${r.traceback.replace(/[<&]/g,
        c => c === "<" ? "&lt;" : "&amp;")}</div>` : ""}
    ${player}
  </div>`;
}

function clipCard(g) {
  const open = OPEN.has(g.key);
  const compute = g.runs.reduce((s, r) => s + (r.elapsed || 0), 0);
  const failed = g.runs.filter(r => r.state === "failed").length;
  const named = g.key.startsWith("\\u0000") ? "not clip-scoped" : g.key;
  const withShot = [...g.runs].reverse().find(r => r.video);
  const sum = [`${g.runs.length} stage${g.runs.length > 1 ? "s" : ""}`,
               `${eta(compute)} compute`, when(g.latest)];
  if (failed) sum.push(`${failed} failed`);
  return `<div class="clip">
    <div class="cliphead" onclick="toggleClip('${g.key.replace(/'/g, "\\\\'")}')">
      ${withShot ? `<img class="clipshot" loading="lazy" alt=""
        src="/thumb?run=${encodeURIComponent(withShot.run)}"
        onerror="this.remove()">` : ""}
      <div style="min-width:0">
        <div class="clipname">${named}</div>
        <div class="clipsum">${sum.join(" · ")}</div>
      </div>
      <span class="caret">${open ? "&minus;" : "+"}</span>
    </div>
    ${open ? `<div class="clipstages">
      ${withShot ? `<div class="srow">
        <div class="srowtop"><span class="srowname">source</span></div>
        <div class="srowtime">${clipName(withShot.video)}</div>
        <video class="art" controls preload="none"
          poster="/thumb?run=${encodeURIComponent(withShot.run)}"
          src="/file?run=${encodeURIComponent(withShot.run)}&what=video"></video>
      </div>` : ""}
      ${g.runs.map(runRow).join("")}</div>` : ""}
  </div>`;
}

function toggleClip(key) {
  OPEN.has(key) ? OPEN.delete(key) : OPEN.add(key);
  historySig = "";              // force one re-render
  renderHistory(LAST_HISTORY);
}

let LAST_HISTORY = [];
function renderHistory(runs) {
  LAST_HISTORY = runs;
  const sig = runs.map(r => r.run).join(",") + "|" + [...OPEN].join(",");
  if (sig === historySig) return;
  historySig = sig;
  const groups = groupRuns(runs);
  document.getElementById("history").innerHTML = groups.length
    ? groups.map(clipCard).join("")
    : '<div class="empty">(no runs archived yet)</div>';
}

// --- Views -------------------------------------------------------------------
//
// Default to whichever view has something in it: on a machine where nothing is
// running -- which is most of the time -- History is the page worth landing on.

let VIEW = null;                 // null until the user picks; then it sticks
function setView(v) { VIEW = v; applyView(); }

let liveCount = 0, histCount = 0;
function applyView() {
  const v = VIEW || (liveCount ? "live" : "history");
  document.getElementById("jobs").hidden = v !== "live";
  document.getElementById("history").hidden = v !== "history";
  document.getElementById("tab-live").classList.toggle("on", v === "live");
  document.getElementById("tab-history").classList.toggle("on", v === "history");
  document.getElementById("n-live").textContent = liveCount || "";
  document.getElementById("n-history").textContent = histCount || "";
}

async function tick() {
  try {
    const [jobs, runs] = await Promise.all([
      (await fetch("/jobs", {cache: "no-store"})).json(),
      (await fetch("/history", {cache: "no-store"})).json(),
    ]);
    document.getElementById("clock").textContent = new Date().toLocaleTimeString();
    const rest = jobs.filter(j => stageOf(j.job) < 0);
    const sections = STAGES.map(([name, desc], i) =>
      stageSection(i + 1, name, desc,
                   jobs.filter(j => stageOf(j.job) === i),
                   i === STAGES.length - 1 && !rest.length));
    if (rest.length) sections.push(stageSection("+", "Other",
      "Jobs this page does not recognise yet.", rest, true));
    document.getElementById("jobs").innerHTML = sections.join("");
    liveCount = jobs.filter(j => j.state === "running" && j.silent <= STALE_S).length;
    histCount = new Set(runs.map(r => clipName(r.video) || "-")).size;
    renderHistory(runs);
    applyView();
  } catch (e) { /* server restarting; keep polling */ }
}
tick();
setInterval(tick, 1500);
</script>
"""


def _under_root(raw):
    """Resolve a path from a job file, refusing anything outside the repo.

    The server only listens on 127.0.0.1 and the paths come from files this
    code wrote, but it opens folders and reads video, so it checks anyway.
    """
    if not raw:
        return None
    p = Path(raw)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    try:
        p.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return p


_DURATIONS = {}
_THUMBS = {}


def _duration(video):
    key = str(video)
    if key not in _DURATIONS:
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", key],
                capture_output=True, text=True, timeout=10)
            _DURATIONS[key] = float(r.stdout.strip() or 0)
        except (OSError, ValueError, subprocess.SubprocessError):
            _DURATIONS[key] = 0.0
    return _DURATIONS[key]


def _thumbnail(video, fraction):
    """One frame from where the run has reached, as JPEG bytes."""
    import subprocess
    dur = _duration(video)
    at = max(0.0, min(fraction, 0.98)) * dur if dur else 0.0
    key = (str(video), round(at, 1))
    if key in _THUMBS:
        return _THUMBS[key]
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", "scale=336:-2", "-f", "image2",
             "-c:v", "mjpeg", "pipe:1"],
            capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode or not r.stdout:
        return None
    if len(_THUMBS) > 80:          # a long session should not grow forever
        _THUMBS.clear()
    _THUMBS[key] = r.stdout
    return r.stdout


def serve(port):
    import http.server
    from urllib.parse import parse_qs, urlparse

    try:                            # winget put ffmpeg on PATH after this shell
        import config               # started; rebuild it or the thumbnails die
        config.ensure_ffmpeg()
    except Exception:
        pass

    def job_named(name):
        for d in read_jobs():
            if d["job"] == name:
                return d
        return None

    def run_named(rid):
        for d in read_history():
            if d["run"] == rid:
                return d
        return None

    def lookup(q):
        """A live job or an archived run, whichever the query names."""
        rid = q.get("run", [""])[0]
        return run_named(rid) if rid else job_named(q.get("job", [""])[0])

    class Handler(http.server.BaseHTTPRequestHandler):
        def _empty(self, code):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _serve_file(self, path, ctype):
            """Byte-range aware, because a <video> cannot seek without it."""
            size = path.stat().st_size
            rng = self.headers.get("Range", "")
            start, end = 0, size - 1
            partial = False
            if rng.startswith("bytes="):
                lo, _, hi = rng[6:].partition("-")
                try:
                    if lo:
                        start = int(lo)
                        end = int(hi) if hi else end
                    elif hi:                      # bytes=-N, the last N bytes
                        start = max(0, size - int(hi))
                    partial = 0 <= start <= end < size
                except ValueError:
                    partial = False
            if not partial:
                start, end = 0, size - 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            remaining = end - start + 1
            with open(path, "rb") as fh:
                fh.seek(start)
                while remaining > 0:
                    chunk = fh.read(min(262144, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    try:
                        self.wfile.write(chunk)
                    except OSError:           # the player seeked away; normal
                        return

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/thumb":
                d = lookup(parse_qs(u.query))
                video = _under_root(d.get("video")) if d else None
                frac = (d["done"] / d["total"]
                        if d and d.get("total") and d["done"] else 0.0)
                jpeg = _thumbnail(video, frac) if video and video.exists() else None
                if not jpeg:
                    return self._empty(404)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if u.path == "/file":
                d = lookup(parse_qs(u.query))
                which = parse_qs(u.query).get("what", ["artifact"])[0]
                target = _under_root((d or {}).get(
                    "video" if which == "video" else "artifact"))
                if not target or not target.is_file():
                    return self._empty(404)
                ctype = {".mp4": "video/mp4", ".webm": "video/webm",
                         ".jpg": "image/jpeg", ".png": "image/png",
                         ".json": "application/json"}.get(
                             target.suffix.lower(), "application/octet-stream")
                return self._serve_file(target, ctype)
            if u.path == "/jobs":
                body = json.dumps(read_jobs()).encode()
                ctype = "application/json"
            elif u.path == "/history":
                # Sweep on every read, not just at startup. A job launched
                # before this module was last edited holds the old code in a
                # live interpreter and will never archive itself; so does any
                # run killed at the terminal. Twelve small reads a poll is
                # cheaper than an archive with holes in it.
                seed_history()
                body = json.dumps(read_history()).encode()
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

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/open":
                job = parse_qs(u.query).get("job", [""])[0]
                d = job_named(job) if job else None
                target = _under_root((d or {}).get("folder") or "out")
                if target and target.exists():
                    if target.is_file():
                        target = target.parent
                    if hasattr(os, "startfile"):
                        os.startfile(target)          # Windows Explorer
                    else:
                        import subprocess
                        subprocess.Popen(
                            ["open" if sys.platform == "darwin" else "xdg-open",
                             str(target)])
            elif u.path == "/dismiss":
                job = parse_qs(u.query).get("job", [""])[0]
                # the job name came from a filename we wrote; still, never let
                # a request walk the filesystem
                target = (DIR / f"{job}.json").resolve()
                if job and target.parent == DIR.resolve() and target.exists():
                    target.unlink()
            elif u.path == "/clear-finished":
                for f in DIR.glob("*.json"):
                    try:
                        if json.loads(f.read_text())["state"] in ("done", "failed"):
                            f.unlink()
                    except (json.JSONDecodeError, OSError, KeyError):
                        pass
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    DIR.mkdir(parents=True, exist_ok=True)
    seeded = seed_history()
    if seeded:
        print(f"archived {seeded} finished job(s) that predate history", flush=True)
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
