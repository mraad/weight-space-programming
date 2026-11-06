#!/usr/bin/env python3
"""Ask local models the same tool-free questions as Claude, many times each, and summarize the scores.

Usage:
  local/sweep.py run --model Qwen/Qwen3.8-27B --urls http://127.0.0.1:8001,http://127.0.0.1:8002 --task lander-pads --replies 20
  local/sweep.py summary

A sweep can restart: it skips each reply seed that already has an answer or a cut-off. A failed call is asked again.
"""
import argparse
import concurrent.futures
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "results" / "local"
TASKS = {"lander-pads": ["lander", "pads"], "exp1-breakout": ["exp1", "breakout"],
         "exp1-breakout-shuffle": ["exp1", "breakout", "--shuffle"], "exp5-breakout": ["exp5", "breakout"]}
USABLE_LANDED = 0.9


def task_name(record):
    """A run with other request options, such as a bigger token cap, is a separate condition and never merges with the baseline."""
    base = f"{record['exp']}-{record['item']}{'-shuffle' if record.get('shuffle') else ''}"
    return f"{base} {record['extra_body']}" if record.get("extra_body") else base


def load_runs(root=LOCAL):
    runs = []
    for path in sorted(root.glob("*/run.json")) if root.exists() else []:
        score = path.with_name("score.json")
        runs.append({"dir": path.parent, "record": json.loads(path.read_text()),
                     "score": json.loads(score.read_text()) if score.exists() else None})
    return runs


def outcome(run):
    """Every reply lands in one group. A cut-off is the model's result; an error is the server's, so it does not count."""
    audit = run["record"]["audit"]
    if not audit["tool_free"] and audit.get("tools_offered") is None:
        return "error"  # no event stream came back
    if not audit["tool_free"]:
        return "tool_call"
    if audit["stop_reason"] != "end_turn":
        return "cut_off"
    return "answered" if run["score"] else "unscored"


def seeds_left(runs, model, task, replies, first=0):
    done = {r["record"].get("reply_seed") for r in runs
            if r["record"]["model"] == model and task_name(r["record"]) == task and outcome(r) in ("answered", "cut_off", "tool_call")}
    return [seed for seed in range(first, replies) if seed not in done]


def one_reply(args, url, seed):
    """One subprocess per reply: wsp.py names each run folder by process id, so threads in one process would collide."""
    python, wsp = sys.executable, str(ROOT / "wsp.py")
    extra = ["--extra-body", args.extra_body] if args.extra_body else []
    asked = subprocess.run([python, wsp, "ask", *TASKS[args.task], "--base-url", url, "--model", args.model, "--reply-seed", str(seed),
                            "--timeout", str(args.timeout), *extra], capture_output=True, text=True)
    run = asked.stdout.splitlines()[0] if asked.stdout else ""
    if asked.returncode:  # a cut-off or a failed call: ask refuses it and it is counted, not scored
        print(f"seed {seed} not scored: {asked.stderr.strip()[:300]}", flush=True)
        return
    scored = subprocess.run([python, wsp, "score", run], capture_output=True, text=True)
    print(f"seed {seed}: {run}" if not scored.returncode else f"seed {seed} scoring failed: {scored.stderr.strip()[:300]}", flush=True)


def sweep(args):
    if not shutil.which("node"):  # the scorer needs it, and a reply can take most of an hour
        sys.exit("node is not on PATH, so replies could not be scored")
    urls = args.urls.split(",")
    condition = f"{args.task} {args.extra_body}" if args.extra_body else args.task
    seeds = seeds_left(load_runs(), args.model, condition, args.replies, args.first_seed)
    print(f"{args.model} {args.task}: {len(seeds)} of {args.replies} replies to ask", flush=True)

    def worker(url):
        while True:
            try:
                seed = seeds.pop(0)
            except IndexError:
                return
            one_reply(args, url, seed)

    with concurrent.futures.ThreadPoolExecutor(len(urls) * args.per_server) as pool:
        for future in [pool.submit(worker, url) for url in urls * args.per_server]:
            future.result()
    print(render(summarize(load_runs())))


def task_metric(runs):
    scores = [r["score"] for r in runs]
    if scores[0].get("eval", {}).get("fixed_target"):
        landed = [s["eval"]["fixed_target"]["landed"] for s in scores]
        return {"usable": sum(rate >= USABLE_LANDED for rate in landed), "landed_median": statistics.median(landed), "landed_max": max(landed),
                "landed_switch_max": max(s["eval"]["mid_flight_switch"]["landed"] for s in scores)}
    if scores[0].get("eval"):
        means = [s["eval"]["mean"] for s in scores]
        return {"score_median": statistics.median(means), "score_max": max(means)}
    confidence = [s["confidence"] for s in scores if s.get("correct") and s["confidence"] is not None]
    return {"correct": sum(bool(s.get("correct")) for s in scores), "correct_confidence_median": statistics.median(confidence) if confidence else None}


def summarize(runs):
    rows = []
    for key in sorted({(r["record"]["model"], task_name(r["record"])) for r in runs}):
        group = [r for r in runs if (r["record"]["model"], task_name(r["record"])) == key]
        counted = [r for r in group if outcome(r) != "error"]
        answered = [r for r in counted if outcome(r) == "answered"]
        tokens = [r["record"]["audit"]["usage"].get("output_tokens") for r in counted]
        tokens = [t for t in tokens if t is not None]
        row = {"model": key[0], "task": key[1], "replies": len(counted), "answered": len(answered),
               "cut_off": sum(outcome(r) == "cut_off" for r in counted), "tool_call": sum(outcome(r) == "tool_call" for r in counted),
               "unscored": sum(outcome(r) == "unscored" for r in counted), "server_errors": len(group) - len(counted),
               # Repeats mean the sample is smaller than it looks.
               "distinct_answers": len({(r["dir"] / "answer.md").read_text() for r in answered}),
               "output_tokens_median": statistics.median(tokens) if tokens else None}
        if answered:
            edits = [r["score"] for r in answered if "block" in r["score"]]
            if edits:
                row["parsed"] = sum(s["block"] and s["applied"] > 0 for s in edits)
            row.update(task_metric(answered))
        rows.append(row)
    return rows


def render(rows):
    columns = ["model", "task", "replies", "answered", "cut_off", "tool_call", "unscored", "server_errors", "distinct_answers", "parsed", "usable",
               "landed_median", "landed_max", "landed_switch_max", "score_median", "score_max", "correct", "correct_confidence_median", "output_tokens_median"]
    used = [c for c in columns if any(row.get(c) is not None for row in rows)]
    cell = lambda v: "" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(used) + " |", "|" + "---|" * len(used)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in used) + " |" for row in rows]
    return "\n".join(lines)


def summary(_args):
    rows = summarize(load_runs())
    (LOCAL / "summary.json").write_text(json.dumps(rows, indent=1))
    print(render(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--urls", required=True, help="comma-separated servers that all hold this model")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--replies", type=int, default=20)
    p.add_argument("--per-server", type=int, default=1, help="replies in flight on each server")
    p.add_argument("--first-seed", type=int, default=0, help="skip lower seeds, such as a reply still running elsewhere")
    p.add_argument("--timeout", type=int, default=14400)
    p.add_argument("--extra-body", help="JSON request options; the summary shows them as a separate condition")
    p.set_defaults(func=sweep)
    sub.add_parser("summary").set_defaults(func=summary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
