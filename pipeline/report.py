"""Grade a finished segment, without needing anyone to label it first.

Every defect this project shipped was found by a human watching the render:
two labels on one man, a name on the wrong player, a spectator in a jersey
taking a prompt, a gate deleting a correct answer. Watching is the most
expensive way to find a bug and the least reliable -- the label that sat on
Towns for two fifths of a clip survived two days of cropped frames.

Most of those defects break a rule of basketball, and rules of basketball can
be checked without a ground truth:

    ten players on the floor          labels drawn per frame should be ten
    five a side                       distinct (club, number) per club
    one man, one identity             no (club, number) on two live tracks
    one identity, one label           no player drawn twice in a frame

None of these need a truth file, so they run on any clip the moment it
finishes, and they are what makes it possible to rank candidate segments
before deciding which is worth labelling. Measured against the four segments
that do have truth, labels-per-frame over ten tracks real coverage closely
(10.00 -> 100%, 9.98 -> 99.8%, 9.33 -> 91.7%) with one caveat that matters:
it counts labels without checking them, so it is an upper bound. A run that
draws a wrong name scores well here and badly against truth. Use this to
shortlist, `score.py` to decide.

    python pipeline/report.py --video out/segments/X.mp4      # one segment
    python pipeline/report.py --index                          # all of them

Writes out/report_<stem>.html (self-contained, opens in a browser) and
out/report_<stem>.json for programmatic comparison.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import overlap
from score import drawn

ON_COURT = 10


def analyse(tracks_path, identities_path, truth_path=None):
    """Every number this report shows, computed in one place."""
    tr = json.loads(Path(tracks_path).read_text())
    doc = json.loads(Path(identities_path).read_text())
    frames = {int(k): v for k, v in tr["frames"].items()}
    idn = {int(k): v for k, v in doc["identities"].items()}
    collapse = {int(k): [tuple(s) for s in v]
                for k, v in (doc.get("overlap") or {}).get("collapse", {}).items()}
    fps = tr.get("fps", 59.94)
    order = sorted(frames)

    # Players, not tracks. Two halves of a merged track both carry the name,
    # and counting tracks made a five-a-side clip look like nine a side.
    players = {}
    for tid, v in idn.items():
        if v.get("name") and not v.get("ignored"):
            players.setdefault((v.get("club"), v["number"]),
                               {"name": v["name"], "tracks": []})
            players[(v.get("club"), v["number"])]["tracks"].append(tid)
    by_club = Counter(club for club, _ in players)

    per_frame, boxes_per_frame, doubles = [], [], 0
    seen_by_player = defaultdict(set)
    for f in order:
        boxes_per_frame.append(len(frames[f]))
        rows = drawn(frames, idn, collapse, f)
        per_frame.append(len(rows))
        here = Counter()
        for r in rows:
            v = idn.get(r["tid"]) or {}
            key = (v.get("club"), v.get("number"))
            here[key] += 1
            seen_by_player[key].add(f)
        doubles += sum(1 for k, n in here.items() if n > 1)

    # (club, number) claimed by tracks that are alive at the same time
    clashes = []
    life = overlap.lifetimes(frames)
    for key, p in players.items():
        for i in range(len(p["tracks"])):
            for j in range(i + 1, len(p["tracks"])):
                a, b = life[p["tracks"][i]], life[p["tracks"][j]]
                shared = min(a[1], b[1]) - max(a[0], b[0]) + 1
                if shared > fps * 0.5:
                    clashes.append((key, p["tracks"][i], p["tracks"][j],
                                    round(shared / fps, 2)))

    gates = Counter()
    for v in idn.values():
        if v.get("ignored"):
            gates[v["ignored"]] += 1
    spans = [{"pair": list(o["pair"]), "kind": o["kind"],
              "from": round(o["start"] / fps, 2), "to": round(o["end"] / fps, 2)}
             for o in []]

    n = len(order)
    checks = [
        ("five a side", sorted(by_club.values()) == [5, 5],
         f"{dict(by_club)}"),
        # the count matters as much as the low: one frame at zero on the last
        # frame of a clip is not the same defect as fifteen seconds at eight
        ("ten labels every frame",
         sum(1 for x in per_frame if x < ON_COURT) == 0,
         f"{sum(1 for x in per_frame if x < ON_COURT)} of {len(per_frame)} "
         f"frames short ({sum(1 for x in per_frame if x < ON_COURT)/max(1,len(per_frame)):.1%}), "
         f"low of {min(per_frame or [0])}, mean {sum(per_frame)/max(1,n):.2f}"),
        ("no player drawn twice", doubles == 0, f"{doubles} frames"),
        ("no shared identity", not clashes,
         "; ".join(f"{k[1]} on tracks {a} and {b} for {s}s"
                   for k, a, b, s in clashes) or "none"),
    ]

    out = {
        "stem": Path(tracks_path).stem.replace("_tracks", ""),
        "seconds": round(n / fps, 2), "frames": n, "fps": fps,
        "tracks": len(idn), "players": len(players),
        "by_club": dict(by_club),
        "labels_per_frame": [round(x, 3) for x in per_frame],
        "boxes_per_frame": boxes_per_frame,
        "mean_labels": round(sum(per_frame) / max(1, n), 3),
        "full_lineup_share": round(sum(1 for x in per_frame if x >= ON_COURT)
                                   / max(1, n), 4),
        "proxy_coverage": round(sum(per_frame) / max(1, n) / ON_COURT, 4),
        "gates": dict(gates),
        "checks": [{"name": a, "pass": bool(b), "detail": c} for a, b, c in checks],
        "timeline": {f"{club or '?'}#{num}": sorted(fs)
                     for (club, num), fs in
                     ((k, seen_by_player[k]) for k in players)},
        "names": {f"{club or '?'}#{num}": p["name"]
                  for (club, num), p in players.items()},
    }
    out["verdict"] = "ready" if all(c["pass"] for c in out["checks"]) else "not ready"

    if truth_path and Path(truth_path).exists():
        from score import score as truth_score
        s = truth_score(truth_path, identities_path)
        out["truth"] = {"precision": round(s["precision"], 4),
                        "coverage": round(s["coverage"], 4),
                        "right": s["right"], "wrong": s["wrong"],
                        "unknown": s["unknown"]}
    return out


def sparkline(values, width=880, height=90, top=None, colour="#c8f031"):
    """An inline SVG line, because a chart should not need a CDN."""
    if not values:
        return ""
    top = top or max(max(values), ON_COURT)
    step = width / max(1, len(values) - 1)
    pts = " ".join(f"{i*step:.1f},{height - v/top*height:.1f}"
                   for i, v in enumerate(values))
    y10 = height - ON_COURT / top * height
    return (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
            f'class="chart">'
            f'<line x1="0" y1="{y10:.1f}" x2="{width}" y2="{y10:.1f}" '
            f'class="ten"/>'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2"/></svg>')


def timeline_svg(report, width=880, row=17):
    """One bar per player, gaps where no correct label was drawn."""
    keys = sorted(report["timeline"], key=lambda k: -len(report["timeline"][k]))
    n = report["frames"]
    out = [f'<svg viewBox="0 0 {width} {len(keys)*row + 4}" class="gantt">']
    for i, k in enumerate(keys):
        fs = set(report["timeline"][k])
        y = i * row
        out.append(f'<rect x="0" y="{y}" width="{width}" height="{row-5}" '
                   f'class="gap"/>')
        start = None
        for f in range(n + 1):
            if f in fs and start is None:
                start = f
            elif f not in fs and start is not None:
                out.append(f'<rect x="{start/n*width:.1f}" y="{y}" '
                           f'width="{max(1,(f-start)/n*width):.1f}" '
                           f'height="{row-5}" class="on"/>')
                start = None
    out.append("</svg>")
    return "".join(out)


PAGE = """<!doctype html>
<meta charset="utf-8"><title>courtvision — {stem}</title>
<style>
 body{{margin:0;background:#08090c;color:#eef1f5;
      font:400 14px/1.55 ui-sans-serif,system-ui,sans-serif}}
 .shell{{max-width:960px;margin:0 auto;padding:2rem 1.5rem 4rem}}
 h1{{font-size:20px;margin:0 0 .2rem}} .sub{{color:#6a7383;font-size:12px;
      font-family:ui-monospace,monospace;margin-bottom:1.6rem}}
 .verdict{{display:inline-block;padding:.35rem .8rem;border-radius:999px;
      font:700 12px/1 ui-monospace,monospace;letter-spacing:.08em}}
 .ready{{background:#123830;color:#3bc9a8}} .no{{background:#3d1b1b;color:#ff7373}}
 .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin:1.4rem 0}}
 .card{{background:#0e1015;border:1px solid #161a20;border-radius:.7rem;padding:.8rem}}
 .k{{color:#6a7383;font:500 10px/1 ui-monospace,monospace;letter-spacing:.1em;
      text-transform:uppercase}}
 .v{{font:700 22px/1.3 ui-sans-serif;margin-top:.35rem}}
 h2{{font-size:13px;color:#99a2af;margin:2rem 0 .6rem;font-weight:600;
      text-transform:uppercase;letter-spacing:.1em}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 td,th{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #161a20}}
 th{{color:#6a7383;font:500 11px/1 ui-monospace,monospace;text-transform:uppercase}}
 .ok{{color:#3bc9a8}} .bad{{color:#ff7373}}
 .chart{{width:100%;height:90px;background:#050608;border-radius:.5rem;
      border:1px solid #161a20}}
 .ten{{stroke:#3e4d12;stroke-width:1;stroke-dasharray:4 4}}
 .gantt{{width:100%;background:#050608;border-radius:.5rem;border:1px solid #161a20}}
 .gap{{fill:#1a1d24}} .on{{fill:#c8f031}}
 .names{{font:400 11px/1.6 ui-monospace,monospace;color:#6a7383;
      display:grid;grid-template-columns:repeat(2,1fr);gap:0 1rem;margin-top:.5rem}}
 code{{color:#99a2af}}
</style>
<div class="shell">
 <h1>{stem}</h1>
 <div class="sub">{seconds}s · {frames} frames · {tracks} tracks · {players} players</div>
 <span class="verdict {vcls}">{verdict}</span>
 <div class="grid">
  <div class="card"><div class="k">labels / frame</div><div class="v">{mean_labels}</div></div>
  <div class="card"><div class="k">proxy coverage</div><div class="v">{proxy}</div></div>
  <div class="card"><div class="k">full lineup</div><div class="v">{full}</div></div>
  <div class="card"><div class="k">truth</div><div class="v">{truth}</div></div>
 </div>
 <h2>Rules of basketball</h2>
 <table>{checks}</table>
 <h2>Labels drawn per frame</h2>
 {chart}
 <h2>Who is labelled, and when</h2>
 {gantt}
 <div class="names">{names}</div>
 <h2>What the gates removed</h2>
 <table>{gates}</table>
</div>
"""


def render_html(rep):
    checks = "".join(
        f'<tr><td>{c["name"]}</td>'
        f'<td class="{"ok" if c["pass"] else "bad"}">{"pass" if c["pass"] else "FAIL"}</td>'
        f'<td><code>{c["detail"]}</code></td></tr>' for c in rep["checks"])
    gates = "".join(f'<tr><td>{k}</td><td>{v} tracks</td></tr>'
                    for k, v in sorted(rep["gates"].items())) or \
        '<tr><td colspan="2"><code>nothing removed</code></td></tr>'
    t = rep.get("truth")
    truth = (f'{t["precision"]:.0%} / {t["coverage"]:.0%}' if t else "—")
    names = "".join(f'<div>{k} &nbsp; {v}</div>'
                    for k, v in sorted(rep["names"].items()))
    return PAGE.format(
        stem=rep["stem"], seconds=rep["seconds"], frames=rep["frames"],
        tracks=rep["tracks"], players=rep["players"],
        verdict=rep["verdict"], vcls="ready" if rep["verdict"] == "ready" else "no",
        mean_labels=f'{rep["mean_labels"]:.2f}',
        proxy=f'{rep["proxy_coverage"]:.0%}',
        full=f'{rep["full_lineup_share"]:.0%}',
        truth=truth, checks=checks, gates=gates, names=names,
        chart=sparkline(rep["labels_per_frame"]),
        gantt=timeline_svg(rep))


def paths_for(stem):
    return (ROOT / "out" / f"{stem}_tracks.json",
            ROOT / "out" / f"{stem}_identities.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="segment mp4; its stem names the artifacts")
    ap.add_argument("--tracks")
    ap.add_argument("--identities")
    ap.add_argument("--truth")
    ap.add_argument("--index", action="store_true",
                    help="one page ranking every segment that has a report")
    args = ap.parse_args()

    out_dir = ROOT / "out"
    if args.index:
        reports = []
        for f in sorted(out_dir.glob("report_*.json")):
            reports.append(json.loads(f.read_text()))
        reports.sort(key=lambda r: (-r["proxy_coverage"], -r["seconds"]))
        rows = "".join(
            f'<tr><td><a href="report_{r["stem"]}.html">{r["stem"]}</a></td>'
            f'<td>{r["seconds"]}s</td><td>{r["players"]}</td>'
            f'<td>{r["mean_labels"]:.2f}</td>'
            f'<td>{r["proxy_coverage"]:.0%}</td>'
            f'<td>{(f"{r["truth"]["precision"]:.0%} / {r["truth"]["coverage"]:.0%}") if r.get("truth") else "—"}</td>'
            f'<td class="{"ok" if r["verdict"]=="ready" else "bad"}">{r["verdict"]}</td></tr>'
            for r in reports)
        # What the LLM audits have cost so far. vlm_check appends one line per
        # call to out/llm_calls.jsonl; summing it here is what turns "we call
        # an LLM sometimes" into a number a budget can be planned around.
        calls, spent = 0, 0.0
        ledger = out_dir / "llm_calls.jsonl"
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                calls += 1
                spent += row.get("cost_usd") or 0.0
        llm = (f'<div class="sub">llm audits: {calls} calls, '
               f'${spent:.2f} total (out/llm_calls.jsonl)</div>') if calls else ""
        page = PAGE.split("<div class=\"shell\">")[0] + f"""<div class="shell">
 <h1>segments</h1><div class="sub">ranked by proxy coverage</div>{llm}
 <table><tr><th>segment</th><th>length</th><th>players</th><th>labels/frame</th>
 <th>proxy</th><th>truth p/c</th><th>verdict</th></tr>{rows}</table></div>"""
        (out_dir / "report_index.html").write_text(page, encoding="utf-8")
        print(f"wrote out/report_index.html ({len(reports)} segments)")
        for r in reports:
            print(f"  {r['stem']:<26} {r['proxy_coverage']:>6.0%}  {r['verdict']}")
        if calls:
            print(f"  llm audits: {calls} calls, ${spent:.2f} total")
        return 0

    if args.video and not (args.tracks and args.identities):
        stem = Path(args.video).stem
        tp, ip = paths_for(stem)
    else:
        tp, ip = Path(args.tracks), Path(args.identities)
    # Truth files are named by hand and do not follow the artifact stem, so
    # find the one that was built against these very boxes rather than guessing
    # from the filename.
    truth = args.truth
    if not truth:
        want = Path(tp).as_posix().split("/")[-1]
        for f in sorted((ROOT / "eval").glob("*_truth.json")):
            try:
                if Path(json.loads(f.read_text()).get("boxes", "")).name == want:
                    truth = str(f)
                    break
            except (json.JSONDecodeError, OSError):
                pass

    rep = analyse(tp, ip, truth)
    (out_dir / f"report_{rep['stem']}.json").write_text(json.dumps(rep))
    (out_dir / f"report_{rep['stem']}.html").write_text(render_html(rep),
                                                        encoding="utf-8")
    print(f"{rep['stem']}: {rep['verdict']}")
    for c in rep["checks"]:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"  labels/frame {rep['mean_labels']:.2f}, "
          f"proxy coverage {rep['proxy_coverage']:.0%}")
    if rep.get("truth"):
        print(f"  truth: precision {rep['truth']['precision']:.1%}, "
              f"coverage {rep['truth']['coverage']:.1%}")
    print(f"  wrote out/report_{rep['stem']}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
