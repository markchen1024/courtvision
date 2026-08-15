"""Ask a vision LLM the one question geometry cannot answer: is that the same man?

The prompt-frame gate counts boxes; it cannot tell whether two boxes hold two
players or one player twice. That distinction cost seg_g6206_43s its fifth
Piston: frame 0 detected Hart in two well-separated boxes, both survived
de-duplication, so ten prompts covered nine men and Beasley was never tracked.
The lineup render showed it plainly -- to anyone who looked. Nobody looked
until the finished video was watched.

This automates the looking. `claude -p` (headless Claude Code, the account
already logged in on this machine -- no new key, no new dependency) reads the
lineup image and answers three questions a human answers at a glance: how many
DISTINCT players carry a prompt box, is anyone boxed twice, is anyone on the
floor unboxed. One call, under a minute, against twenty-five minutes of GPU
a bad prompt frame wastes.

Validated before being wired in, on the cases with known answers:

    seg_g6206_43s frame 0   bad_prompt_frame, 9 distinct   HIT (the Hart double)
    seg_02m28.00s_13s       ok, 10 distinct                clean, no false alarm
                            (it also recognised the struck box as "a fan in a
                            Cunningham jersey", unprompted)
    seg_02m44.15s_10s       ok, 10 distinct                clean, no false alarm

Advisory by default: run_segment prints the verdict and carries on, because a
three-case validation set earns a warning light, not a kill switch. --strict
makes a bad verdict fatal. Every verdict lands in out/<stem>_audit.json so the
report can fold it in and the validation set grows with every segment run.

Every call is also a cost, and costs get a ledger. The headless envelope
(--output-format json) reports total_cost_usd, token usage and duration; each
call appends one line to out/llm_calls.jsonl and the report index sums it.
Measured baseline: a trivial no-image call is ~$0.45, almost all of it the
21.6k-token system prompt being written to cache -- so the marginal cost of a
segment audit is the image plus a few hundred tokens, not the headline.

    python pipeline/vlm_check.py --image out/seg_g6206_43s_lineup.jpg
    python pipeline/vlm_check.py --image X_lineup.jpg --strict
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """Read the image {image}. It is frame 0 of a basketball clip with \
detector boxes drawn on players (green = kept as SAM2 prompt, red = struck as \
duplicate). Exactly 10 players should be on court, 5 per team. Audit the \
PROMPTS (green boxes only) and answer in strict JSON, nothing else:
{{"distinct_players_prompted": <int, how many DIFFERENT people have a green box>,
 "same_player_boxed_twice": <true/false>,
 "players_without_any_box": <int, players visible on court with no green box>,
 "verdict": "ok" | "bad_prompt_frame",
 "notes": "<one or two short sentences>"}}"""


def find_claude():
    for name in ("claude", "claude.exe", "claude.cmd"):
        p = shutil.which(name)
        if p:
            return p
    return None


LEDGER = ROOT / "out" / "llm_calls.jsonl"


def _log_call(purpose, image, envelope, verdict):
    """One line per LLM call, because cost estimates need a ledger to sum."""
    import time
    usage = envelope.get("usage") or {}
    models = envelope.get("modelUsage") or {}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": purpose,
            "image": str(image),
            "model": next(iter(models), None),
            "cost_usd": envelope.get("total_cost_usd"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read": usage.get("cache_read_input_tokens"),
            "cache_creation": usage.get("cache_creation_input_tokens"),
            "duration_ms": envelope.get("duration_ms"),
            "verdict": (verdict or {}).get("verdict"),
        }) + "\n")


def audit(image, timeout=300, purpose="prompt-frame-audit"):
    """Run the headless audit. Returns (verdict_dict | None, raw, cost_dict)."""
    exe = find_claude()
    if exe is None:
        return None, "claude CLI not on PATH", {}
    rel = Path(image)
    rel = rel if rel.is_absolute() else ROOT / rel
    proc = subprocess.run(
        [exe, "-p", PROMPT.format(image=rel.as_posix()),
         "--allowedTools", "Read", "--output-format", "json"],
        capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL, cwd=ROOT)
    raw = (proc.stdout or "") + (proc.stderr or "")
    # stdout is one JSON envelope; warnings may precede it
    a = raw.find("{")
    if a < 0:
        return None, raw, {}
    try:
        env = json.loads(raw[a:])
    except json.JSONDecodeError:
        return None, raw, {}
    text = env.get("result") or ""
    va, vb = text.find("{"), text.rfind("}")
    verdict = None
    if va >= 0 and vb > va:
        try:
            verdict = json.loads(text[va:vb + 1])
        except json.JSONDecodeError:
            pass
    cost = {"cost_usd": env.get("total_cost_usd"),
            "duration_ms": env.get("duration_ms"),
            "model": next(iter(env.get("modelUsage") or {}), None)}
    _log_call(purpose, image, env, verdict)
    return verdict, raw, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="the lineup render to audit")
    ap.add_argument("--out", help="verdict JSON; default out/<stem>_audit.json")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on a bad verdict instead of just saying so")
    args = ap.parse_args()

    # Visible on the pipeline page while it runs, and archived with the rest:
    # a thirty-second external call that spends quota should not be the one
    # stage nobody can see.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from progress import Progress
    stem0 = Path(args.image).stem.replace("_lineup", "")
    prog = Progress("vlm-audit", total=1, note=Path(args.image).name,
                    artifact=str(ROOT / "out" / f"{stem0}_audit.json"),
                    meta={"image": str(args.image)})
    verdict, raw, cost = audit(args.image)
    if verdict is None:
        # The audit failing must not kill a run the gate already passed: this
        # stage exists to add information, not to add a failure mode.
        prog.fail(note="no verdict")
        print(f"vlm audit unavailable: {raw.strip()[:200]}")
        return 0

    stem = Path(args.image).stem.replace("_lineup", "")
    out = Path(args.out) if args.out else ROOT / "out" / f"{stem}_audit.json"
    out.write_text(json.dumps({"image": str(args.image), **verdict, **cost},
                              indent=1))

    prog.set(1)
    prog.done(note=verdict.get("verdict") or "?")
    ok = verdict.get("verdict") == "ok"
    print(f"vlm audit: {verdict.get('verdict')} -- "
          f"{verdict.get('distinct_players_prompted')} distinct players prompted"
          + (", same player boxed twice"
             if verdict.get("same_player_boxed_twice") else "")
          + (f", {verdict.get('players_without_any_box')} unboxed"
             if verdict.get("players_without_any_box") else ""))
    print(f"  {verdict.get('notes', '')}")
    if cost.get("cost_usd") is not None:
        print(f"  cost ${cost['cost_usd']:.3f}, "
              f"{(cost.get('duration_ms') or 0)/1000:.0f}s, {cost.get('model')}"
              f" -> ledger out/llm_calls.jsonl")
    print(f"  wrote {out.relative_to(ROOT)}")
    if not ok and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
