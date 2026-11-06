#!/usr/bin/env python3
"""Replicate Weight-Space Programming experiments on Claude.

Uses the networks and prompts from github.com/RikiyaT/lm_understand-logs. Each ask is one tool-free,
one-reply call from an empty directory, so the model cannot reach answers or evaluation data.
"""
import argparse
import ast
import hashlib
import http.client
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

SHA = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
BASE = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{SHA}/"
ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
EXP5 = DATA / "exp5_write_from_scratch"
SYSTEM_PROMPT = "Answer the user's request."
MAX_OUTPUT_TOKENS = 64000

GAMES = ["breakout", "seaquest", "asterix"]
EXPERIMENTS = {
    "exp1": {"dir": "exp1_which_game", "items": ["breakout", "pong", "pacman", "freeway"], "effort": "high",
             "prompt": "runs/{}/prompt.txt", "answer": "runs/{}/answer.md", "label": "Game",
             "keywords": {"breakout": r"breakout|brick|arkanoid", "pong": r"\bpong\b", "pacman": r"pac-?\s?man",
                          "freeway": r"freeway|frogger|road.cross|traffic"}},
    "exp2": {"dir": "exp2_diffusion", "items": ["circle", "square", "triangle"], "effort": "high",
             "prompt": "runs/{}/prompt.txt", "answer": "runs/{}/answer.md", "label": "Images",
             "keywords": {"circle": r"circle|circular|disc|round|oval", "square": r"square|rectang|box",
                          "triangle": r"triang|pyramid|wedge"}},
    "exp5": {"dir": "exp5_write_from_scratch", "items": GAMES, "effort": "xhigh", "episodes": 1000,
             "prompt": "runs/{}/message.txt", "answer": "runs/{}/reply/answer.md"},
    "lander": {"items": ["pads"], "effort": "xhigh", "episodes": 3000},
    "tanks": {"items": ["repair", "repair-shuffled", "write"], "effort": "xhigh"},  # prompts come from geo/tanks.py report
}
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
LANDER = ROOT / "lander"
LANDER_SHAPES = {"W1": (8, 16), "b1": (16,), "W2": (16, 16), "b2": (16,), "W3": (16, 4), "b3": (4,)}
NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
            "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}


# --- data ---------------------------------------------------------------------------------------

def data_path(exp, kind, item):
    cfg = EXPERIMENTS[exp]
    return DATA / cfg["dir"] / cfg[kind].format(item)


def fetch(_args=None):
    paths = ["LICENSE", "exp1_which_game/eval/scores.json", "exp2_diffusion/eval/scores.json",
             "exp5_write_from_scratch/LICENSE-MinAtar", "exp5_write_from_scratch/eval/minatar_eval.json"]
    for cfg in EXPERIMENTS.values():
        for item in cfg["items"] if "dir" in cfg else []:
            paths += [f"{cfg['dir']}/{cfg['prompt'].format(item)}", f"{cfg['dir']}/{cfg['answer'].format(item)}"]
    paths += [f"exp5_write_from_scratch/networks/{g}/hand.npz" for g in GAMES]
    paths += [f"exp5_write_from_scratch/runs/{g}/apply.json" for g in GAMES]
    for rel in paths:
        out = DATA / rel
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE + rel, timeout=60) as resp:
            body = resp.read()
        out.write_bytes(body)
        print(f"{len(body):>9}  {rel}")
    print(f"{len(paths)} files in {DATA} at {SHA[:12]}")


# --- reading: scoring and the control ------------------------------------------------------------

def answer_field(text, label):
    """Value of `Label: value` or `## Label` + next line. Last match wins: the field closes the reply."""
    found = re.findall(rf"(?im)^[#*\s]*{label}\b[*\s]*:?[*\s]*(\S.*)$", text)
    return found[-1].strip() if found else ""


def score_read(exp, text):
    cfg = EXPERIMENTS[exp]
    line = answer_field(text, cfg["label"])
    hits = [(m.start(), name) for name, pat in cfg["keywords"].items() if (m := re.search(pat, line, re.I))]
    digits = re.search(r"\d+", answer_field(text, "Confidence"))
    return {"guess": min(hits)[1] if hits else "other", "answer_line": line,
            "confidence": int(digits.group()) if digits else None}


def shuffle_w1(prompt, seed=0):
    """Negative control: permute the rows of W1, so the prompt's row-to-input mapping is false."""
    lines = prompt.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("W1 (")) + 1
    end = next(i for i in range(start, len(lines)) if not lines[i].strip())
    rows = lines[start:end]
    random.Random(seed).shuffle(rows)
    return "\n".join(lines[:start] + rows + lines[end:])


# --- writing: the `edits` language ---------------------------------------------------------------

def statement_patterns(names):
    """Regexes for one statement and for its start. Names can hold dots, such as segmentation_head.0.weight."""
    tensors = "|".join(map(re.escape, names))
    return re.compile(rf"^({tensors})\[([^\]]*)\]\s*([-+*]?=)\s*(.+)$", re.S), re.compile(rf"^({tensors})\[")


def parse_index(text):
    parts = []
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            parts.append(slice(*(int(p) if p.strip() else None for p in part.split(":"))))
        else:
            parts.append(int(part))
    return tuple(parts)


def split_statements(block, starts):
    """Join list continuation lines. A new statement ends an unbalanced one, so one bad line costs one statement."""
    statements, buffer = [], ""
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if buffer and starts.match(line):
            statements.append(buffer)
            buffer = ""
        buffer = f"{buffer} {line}".strip()
        if buffer.count("[") == buffer.count("]"):
            statements.append(buffer)
            buffer = ""
    if buffer:
        statements.append(buffer)
    return statements


def apply_statement(net, statement, pattern):
    match = pattern.match(statement)
    if not match:
        raise ValueError("unreadable statement")
    name, index, op, rhs = match.groups()
    # literal_eval, never eval: the reply is untrusted model output.
    value = np.array(ast.literal_eval(rhs.strip()), dtype=np.float32)
    if not np.all(np.abs(value) <= 1e6):
        raise ValueError("value outside |v| <= 1e6")
    tensor, idx = net[name], parse_index(index)
    target = tensor[idx]
    if value.ndim and value.shape != np.shape(target):
        raise ValueError(f"shape {value.shape} does not match target {np.shape(target)}")
    tensor[idx] = {"=": value, "*=": target * value, "+=": target + value, "-=": target - value}[op]


def apply_edits(reply, shapes=SHAPES, base=None):
    """Build the network from the LAST edits block. Bad statements are skipped and reported, as the prompt promises.
    The network starts from a copy of base if given, else from zeros."""
    net = {name: tensor.copy() for name, tensor in base.items()} if base else {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
    pattern, starts = statement_patterns(net)
    blocks = list(re.finditer(r"```edits[ \t]*\n(.*?)(?:```|\Z)", reply, re.S))
    report = {"block": bool(blocks), "statements": 0, "applied": 0, "errors": [], "cut_off": False}
    if blocks:
        report["cut_off"] = not blocks[-1].group(0).endswith("```")
        statements = split_statements(blocks[-1].group(1), starts)
        report["statements"] = len(statements)
        for statement in statements:
            try:
                apply_statement(net, statement, pattern)
                report["applied"] += 1
            except (ValueError, SyntaxError, IndexError, TypeError, RecursionError, MemoryError) as err:
                report["errors"].append({"statement": statement[:200], "error": str(err)})
    report["nonzero"] = sum(int(np.count_nonzero(t)) for t in net.values())
    return net, report


# --- writing: MinAtar scoring under the post's protocol ------------------------------------------

def channel_matrix(prompt, game_channels):
    """Read the 6-channel mapping from the prompt the model saw, so scoring and prompt cannot disagree."""
    line = next(l for l in prompt.splitlines() if l.startswith("How the 6 channels"))
    sources = re.findall(r"channel \d \(\w+\) <- ([\w, ]+?)[;.]", line)
    if len(sources) != 6:
        raise ValueError(f"expected 6 channels, read {len(sources)}")
    matrix = np.zeros((len(game_channels), 6), np.int8)
    for out, names in enumerate(sources):
        for name in names.split(", "):
            if name != "nothing":
                matrix[game_channels[name], out] = 1
    return matrix


def encode(state, matrix):
    """Flattening (row, column, channel) in C order gives the prompt's (row * 10 + column) * 6 + channel."""
    return ((state.reshape(100, matrix.shape[0]) @ matrix) > 0).ravel().astype(np.float32)


def play(net, game, prompt, seed, max_steps=10_000):
    from minatar import Environment
    env = Environment(game, sticky_action_prob=0.1)
    env.random = np.random.RandomState(seed)
    env.env.random = np.random.RandomState(seed + 1_000_000)
    env.reset()
    rng = np.random.RandomState(seed + 2_000_000)
    matrix = channel_matrix(prompt, env.env.channels)
    total, steps, done = 0.0, 0, False
    while not done and steps < max_steps:
        h1 = np.tanh(encode(env.state(), matrix) @ net["W1"] + net["b1"])
        h2 = np.tanh(h1 @ net["W2"] + net["b2"])
        logits = h2 @ net["W3"] + net["b3"]
        p = np.exp(logits - logits.max()).cumsum()
        reward, done = env.act(int((p > rng.random_sample() * p[-1]).argmax()))
        total += reward
        steps += 1
    return total, steps


def evaluate(net, game, prompt, episodes):
    jobs = [(net, game, prompt, seed) for seed in range(episodes)]
    with multiprocessing.Pool() as pool:
        played = pool.starmap(play, jobs, chunksize=max(1, episodes // 64))
    scores = np.array([s for s, _ in played])
    return {"mean": float(scores.mean()), "se": float(scores.std(ddof=1) / math.sqrt(episodes)) if episodes > 1 else 0.0,
            "max": float(scores.max()), "mean_length": float(np.mean([n for _, n in played])), "episodes": episodes}


def lander_prompt():
    return (LANDER / "prompt.md").read_text().replace("{{SOURCE}}", (LANDER / "sim.js").read_text().strip())


def lander_eval(pilot, episodes):
    """pilot is a network JSON path, `random` or `controller`. sim.js is the one source of the rules, so scoring runs in Node."""
    done = subprocess.run(["node", str(LANDER / "eval.js"), str(pilot), str(episodes)], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def load_hand(game):
    with np.load(EXP5 / f"networks/{game}/hand.npz") as npz:
        return {name: npz[key] for name, key in NPZ_KEYS.items()}


# --- asking the model ----------------------------------------------------------------------------

def audit_events(events):
    """Same test as the authors': a turn counts as tool-free only if the stream shows no tool at all."""
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    tool_calls = [b.get("name") for e in events if e.get("type") == "assistant"
                  for b in e.get("message", {}).get("content", []) if b.get("type") == "tool_use"]
    result = next((e for e in events if e.get("type") == "result"), {})
    offered, mcp = (init.get("tools", []), init.get("mcp_servers", [])) if init else (None, None)
    return {"tools_offered": offered, "mcp_servers": mcp,
            "tool_calls": tool_calls, "tool_free": offered == [] and not mcp and not tool_calls,
            "is_error": result.get("is_error"), "stop_reason": result.get("stop_reason"),
            "usage": result.get("usage"), "cost_usd": result.get("total_cost_usd"), "answer": result.get("result", "")}


def pick(mapping, *keys):
    return {k: mapping[k] for k in keys if k in mapping}


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def scrub_event(event):
    """Allow-list for what gets saved: results are public, and the raw stream holds machine paths, ids and account limits."""
    kind = event.get("type")
    if kind == "system" and event.get("subtype") == "init":
        return pick(event, "type", "subtype", "model", "tools", "mcp_servers", "slash_commands", "skills",
                    "permissionMode", "claude_code_version")
    if kind == "assistant":
        message = event.get("message", {})
        content = [{k: v for k, v in block.items() if k != "signature"} for block in message.get("content", [])]
        return {"type": "assistant", "message": {**pick(message, "model", "role"), "content": content}}
    if kind == "result":
        return {**pick(event, "type", "subtype", "is_error", "stop_reason", "result", "total_cost_usd", "num_turns", "duration_ms"),
                "usage": pick(event.get("usage", {}), *USAGE_KEYS)}
    return None


def write_events(path, events):
    kept = (scrub_event(event) for event in events)
    path.write_text("".join(json.dumps(event) + "\n" for event in kept if event))


def call_local(prompt, base_url, model, timeout, seed=None, extra=None):
    """One reply from an OpenAI-compatible server, returned as the event stream call_claude gives, so the audit and scoring are shared."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    cmd = ["POST", url, model]
    # No `tools` field: the server offers none. A per-reply seed stops identically seeded servers from repeating a reply.
    body = {"model": model, "max_tokens": MAX_OUTPUT_TOKENS, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], **(extra or {})}
    if "tools" in body:
        raise ValueError("a local run must offer no tools")
    if seed is not None:
        body["seed"] = seed
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            reply = json.load(resp)
        choice, usage = reply["choices"][0], reply.get("usage", {})
        message = choice["message"]
    except TimeoutError:
        return cmd, subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout} s")
    except urllib.error.HTTPError as err:  # the server's reason, such as a prompt longer than its context
        return cmd, subprocess.CompletedProcess(cmd, 1, "", f"HTTP {err.code}: {err.read().decode(errors='replace')[:2000]}")
    except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError) as err:
        return cmd, subprocess.CompletedProcess(cmd, 1, "", f"{type(err).__name__}: {err}")
    text = message.get("content") or ""
    # vLLM and llama.cpp name the thinking field differently, and the name has changed between vLLM versions.
    thinking = message.get("reasoning") or message.get("reasoning_content") or ""
    calls = [{"type": "tool_use", "name": call.get("function", {}).get("name")} for call in message.get("tool_calls") or []]
    content = [{"type": "thinking", "thinking": thinking}] * bool(thinking) + [{"type": "text", "text": text}] + calls
    stop = {"stop": "end_turn", "length": "max_tokens"}.get(choice.get("finish_reason"), choice.get("finish_reason"))
    events = [{"type": "system", "subtype": "init", "model": model, "tools": [], "mcp_servers": []},
              {"type": "assistant", "message": {"model": model, "role": "assistant", "content": content}},
              {"type": "result", "subtype": "success", "is_error": False, "stop_reason": stop, "result": text,
               "duration_ms": int((time.time() - started) * 1000),
               "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}}]
    return cmd, subprocess.CompletedProcess(cmd, 0, "".join(json.dumps(event) + "\n" for event in events), "")


def call_claude(prompt, model, effort, timeout):
    # Not --bare: it skips the keychain login. These flags were probed to keep CLAUDE.md, memory, hooks and MCP out;
    # Claude Code's short environment block and account-email reminder remain (README, "Isolation").
    cmd = ["claude", "-p", "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
           "--disable-slash-commands", "--system-prompt", SYSTEM_PROMPT,
           "--output-format", "stream-json", "--verbose", "--model", model, "--effort", effort]
    env = {**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS), "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
           "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}
    with tempfile.TemporaryDirectory() as cwd:
        try:
            done = subprocess.run(cmd, input=prompt, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as err:
            text = lambda out: out.decode(errors="replace") if isinstance(out, bytes) else out or ""
            done = subprocess.CompletedProcess(cmd, 124, text(err.stdout), f"{text(err.stderr)}timed out after {timeout} s")
    return cmd, done


# Every enabled Codex tool feature. Codex still names some tools to the model, but a probe showed that exec and
# spawn_agent both fail closed with these off. No flag removes the names, so an attempt must fail the audit instead.
CODEX_DISABLED = ("apps", "browser_annotation_api", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
                  "code_mode_host", "code_mode_interrupt", "computer_use", "goals", "hooks", "image_generation", "in_app_browser",
                  "multi_agent", "plugins", "remote_plugin", "shell_snapshot", "shell_tool", "skill_mcp_dependency_install",
                  "skill_search", "sleep_tool", "tool_call_mcp_elicitation", "tool_suggest", "unified_exec", "unified_exec_tty",
                  "view_image", "workspace_dependencies", "worktrees", "write_stdin_approval")
CODEX_ITEMS = {"agent_message", "reasoning"}
CODEX_CLOSED = "code-mode host is disabled"  # expected at startup: it is the proof that tool code cannot run


def parse_codex(stdout, stderr, model, wall_s):
    """Turn a Codex JSON stream into the event stream call_claude gives. A tool attempt shows only in stderr, so read both."""
    items, usage, failure, done = [], {}, None, False
    for line in filter(str.strip, stdout.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "item.completed":
            items.append(event.get("item", {}))
        elif kind == "turn.completed":
            usage, done = event.get("usage", {}), True
        elif kind in ("turn.failed", "error"):
            failure = json.dumps(event)[:500]
    text = "\n\n".join(i.get("text", "") for i in items if i.get("type") == "agent_message")
    calls = [i.get("type", "unknown") for i in items if i.get("type") not in CODEX_ITEMS and not
             (i.get("type") == "error" and "Code Mode is unavailable" in i.get("message", ""))]
    calls += ["tool_router_error" for line in stderr.splitlines() if "tools::router" in line]
    content = [{"type": "text", "text": text}] + [{"type": "tool_use", "name": name} for name in calls]
    return [{"type": "system", "subtype": "init", "model": model, "tools": [], "mcp_servers": []},
            {"type": "assistant", "message": {"model": model, "role": "assistant", "content": content}},
            {"type": "result", "subtype": "success", "is_error": bool(failure), "stop_reason": "end_turn" if done and not failure else "not_finished",
             "result": text, "duration_ms": int(wall_s * 1000),
             "usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}}]


def call_codex(prompt, model, effort, timeout):
    """One reply through the Codex login, from an empty folder and an empty Codex home that holds only a link to the login."""
    cmd = ["codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
           "--sandbox", "read-only", "-m", model, "-c", f"model_reasoning_effort={effort}", "-c", "web_search=disabled",
           "--color", "never"]
    for feature in CODEX_DISABLED:
        cmd += ["--disable", feature]
    real_auth = Path.home() / ".codex" / "auth.json"
    started = time.time()
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as cwd:
        link = Path(home) / "auth.json"
        link.symlink_to(real_auth)
        env = {**{k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}, "CODEX_HOME": home}
        try:
            done = subprocess.run(cmd, input=prompt, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as err:
            text = lambda out: out.decode(errors="replace") if isinstance(out, bytes) else out or ""
            done = subprocess.CompletedProcess(cmd, 124, text(err.stdout), f"{text(err.stderr)}timed out after {timeout} s")
        finally:
            if link.is_file() and not link.is_symlink():  # a token refresh replaced the link; losing it would break the user's login
                shutil.copy2(link, real_auth)
    events = parse_codex(done.stdout, done.stderr, model, time.time() - started)
    return cmd, subprocess.CompletedProcess(cmd, done.returncode, "".join(json.dumps(e) + "\n" for e in events), done.stderr)


def parse_events(stdout):
    """Split CLI output into events and other lines. A stray warning must not cost us a reply that was already paid for."""
    events, other = [], []
    for line in filter(str.strip, stdout.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        (events if isinstance(event, dict) else other).append(event if isinstance(event, dict) else line)
    return events, other


def run_problem(record):
    """Why a run cannot count as the model's work, or None. A failed call must never read as a bad reply."""
    audit = record["audit"]
    if not audit["tool_free"]:
        return "the run is not tool-free"
    if record["rc"]:
        return f"the CLI exited with code {record['rc']}"
    if audit["is_error"]:
        return "the call returned an error"
    if audit["stop_reason"] != "end_turn":
        return f"the reply did not finish (stop_reason={audit['stop_reason']})"
    return None


def ask(args):
    if args.exp == "tanks":
        prompt = (RESULTS / "geo" / "prompts" / f"{args.item}.txt").read_text()
    else:
        prompt = lander_prompt() if args.exp == "lander" else data_path(args.exp, "prompt", args.item).read_text()
    if args.shuffle:
        prompt = shuffle_w1(prompt, args.seed)
    # A local server has no effort setting, so a local run must not record one.
    effort = "default" if args.base_url else args.effort or EXPERIMENTS[args.exp]["effort"]
    tag = f"{args.exp}-{args.item}{'-shuffle' if args.shuffle else ''}-{args.model.replace('/', '-')}-{effort}-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    # In their own folder: docs/build.js uses every results/lander-pads-* run, and these are not part of the manual.
    # Tank runs hold numbers from private imagery, so results/geo stays out of git.
    run = (RESULTS / "geo" if args.exp == "tanks" else RESULTS / "local" if args.base_url else RESULTS) / tag
    run.mkdir(parents=True)
    (run / "prompt.txt").write_text(prompt)
    started = time.time()
    if args.base_url:
        cmd, done = call_local(prompt, args.base_url, args.model, args.timeout, args.reply_seed, json.loads(args.extra_body or "{}"))
    elif args.model.startswith("gpt-"):
        cmd, done = call_codex(prompt, args.model, effort, args.timeout)
    else:
        cmd, done = call_claude(prompt, args.model, effort, args.timeout)
    events, other = parse_events(done.stdout)
    (run / "stderr.log").write_text("\n".join([done.stderr, *other]).strip().replace(str(Path.home()), "~"))
    write_events(run / "events.jsonl", events)
    audit = audit_events(events)  # audits the full stream; only the filtered copy is saved
    audit["usage"] = pick(audit["usage"] or {}, *USAGE_KEYS)
    (run / "answer.md").write_text(audit.pop("answer"))
    record = {"exp": args.exp, "item": args.item, "shuffle": args.shuffle, "seed": args.seed if args.shuffle else None,
              "model": args.model, "effort": effort, "reply_seed": args.reply_seed, "extra_body": args.extra_body, "cmd": cmd, "rc": done.returncode, "wall_s": round(time.time() - started, 1),
              "logs_sha": SHA, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "audit": audit}
    (run / "run.json").write_text(json.dumps(record, indent=1))
    print(run)
    print(json.dumps({k: audit[k] for k in ("tool_free", "is_error", "stop_reason", "usage")}, indent=1))
    problem = run_problem(record)
    if problem:
        sys.exit(f"run not usable: {problem}. stderr: {done.stderr[-500:]}")


def score(args):
    run = Path(args.run)
    record = json.loads((run / "run.json").read_text())
    problem = run_problem(record)
    if problem:
        sys.exit(f"refusing to score: {problem}")
    answer, exp, item = (run / "answer.md").read_text(), record["exp"], record["item"]
    if exp == "tanks":
        sys.exit(f"score this run on gc1: geo/tanks.py score {run}")
    episodes = args.episodes or EXPERIMENTS[exp].get("episodes")
    if exp == "lander":
        net, result = apply_edits(answer, LANDER_SHAPES)
        path = run / "network.json"
        path.write_text(json.dumps({k: v.tolist() for k, v in net.items()}))
        result["eval"] = lander_eval(path, episodes)
    elif exp == "exp5":
        net, result = apply_edits(answer)
        # From data/, not the run folder: run prompts quote GPL source and are not committed.
        result["eval"] = evaluate(net, item, data_path(exp, "prompt", item).read_text(), episodes)
        result["published_hand"] = json.loads((EXP5 / "eval/minatar_eval.json").read_text())[item]["hand"]
    else:
        result = score_read(exp, answer)
        result.update(true=item, shuffle=record["shuffle"], correct=result["guess"] == item)
    (run / "score.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def lander_export(args):
    """Write lander/networks.js from scored runs, so the page works from file:// with no server."""
    networks = []
    for name, run in zip("ABCDEFGH", map(Path, sorted(args.runs))):
        scored = json.loads((run / "score.json").read_text())
        networks.append({"name": f"Claude reply {name}",
                         "landed": scored["eval"]["fixed_target"]["landed"],
                         "landedSwitch": scored["eval"]["mid_flight_switch"]["landed"],
                         "design": (run / "answer.md").read_text().split("```edits")[0].strip(),
                         "net": json.loads((run / "network.json").read_text())})
    baselines = {name: lander_eval(name, args.episodes) for name in ("random", "controller")}
    (LANDER / "networks.js").write_text("globalThis.NETWORKS = " + json.dumps({"networks": networks, "baselines": baselines}) + ";\n")
    print(f"{len(networks)} networks written to {LANDER / 'networks.js'}")


def verify(args):
    """Prove the pipeline with no LLM: reproduce the authors' scoring from their own replies and networks."""
    failures, scores = [], {}

    def check(ok, message):
        print(("ok    " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    for exp in ("exp1", "exp2"):
        for row in json.loads((DATA / EXPERIMENTS[exp]["dir"] / "eval/scores.json").read_text()):
            mine = score_read(exp, data_path(exp, "answer", row["true"]).read_text())
            check(mine["guess"] == row["guess"] and mine["confidence"] == row["confidence"],
                  f"{exp} {row['true']}: guess {mine['guess']}, confidence {mine['confidence']} (published {row['guess']}, {row['confidence']})")
    published = json.loads((EXP5 / "eval/minatar_eval.json").read_text())
    for game in GAMES:
        net, report = apply_edits(data_path("exp5", "answer", game).read_text())
        hand = load_hand(game)
        expected = json.loads((EXP5 / f"runs/{game}/apply.json").read_text())
        check(all(np.array_equal(net[k], hand[k]) for k in SHAPES), f"exp5 {game}: parsed reply equals hand.npz exactly")
        check((report["statements"], report["applied"], report["nonzero"]) ==
              (expected["edit_report"]["statements"], expected["edit_report"]["applied"], expected["nonzero_total"]),
              f"exp5 {game}: {report['applied']}/{report['statements']} statements, {report['nonzero']} nonzero")
        if args.episodes:
            result, ref = evaluate(hand, game, data_path("exp5", "prompt", game).read_text(), args.episodes), published[game]["hand"]
            gap = abs(result["mean"] - ref["mean"]) / math.hypot(result["se"], ref["se"])
            check(gap < 3, f"exp5 {game}: score {result['mean']:.1f} ± {result['se']:.1f} vs published {ref['mean']:.1f} ± {ref['se']:.1f} ({gap:.2f} SE)")
            scores[game] = {"mean": result["mean"], "se": result["se"], "published_mean": ref["mean"], "published_se": ref["se"], "gap_se": gap}
    if scores and args.episodes == EXPERIMENTS["exp5"]["episodes"]:  # the manual reads this file, so a short run must not replace it
        (RESULTS / "verify.json").write_text(json.dumps({"episodes": args.episodes, "games": scores}, indent=1))
    sys.exit(f"{len(failures)} checks failed" if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    sub.add_parser("fetch", help="download the authors' logs at the pinned commit").set_defaults(func=fetch)

    p = sub.add_parser("ask", help="one tool-free reply from Claude (spends tokens) or from a local server")
    p.add_argument("exp", choices=EXPERIMENTS)
    p.add_argument("item")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--base-url", help="ask this OpenAI-compatible server, not Claude; the run goes to results/local/")
    p.add_argument("--reply-seed", type=int, help="sampling seed for a local reply")
    p.add_argument("--extra-body", help="JSON request options for a local reply, such as reasoning_effort or max_tokens")
    p.add_argument("--effort", help="defaults to the effort the authors used for this experiment")
    p.add_argument("--shuffle", action="store_true", help="negative control: permute the rows of W1")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p.add_argument("--timeout", type=int, default=3600)
    p.set_defaults(func=ask)

    p = sub.add_parser("score", help="score a run folder")
    p.add_argument("run")
    p.add_argument("--episodes", type=int, help="defaults to the count this experiment is scored at")
    p.set_defaults(func=score)

    p = sub.add_parser("verify", help="offline proof against the authors' replies and networks")
    p.add_argument("--episodes", type=int, default=EXPERIMENTS["exp5"]["episodes"], help="0 skips the game evaluation")
    p.set_defaults(func=verify)

    p = sub.add_parser("lander-export", help="write lander/networks.js from scored lander runs")
    p.add_argument("runs", nargs="*")
    p.add_argument("--episodes", type=int, default=EXPERIMENTS["lander"]["episodes"])
    p.set_defaults(func=lander_export)

    args = parser.parse_args()
    if getattr(args, "exp", None) and args.item not in EXPERIMENTS[args.exp]["items"]:
        parser.error(f"item must be one of {EXPERIMENTS[args.exp]['items']}")
    if args.func is ask and args.shuffle and args.exp in ("exp5", "lander", "tanks"):
        parser.error("--shuffle applies to the read experiments only")
    args.func(args)


if __name__ == "__main__":
    main()
