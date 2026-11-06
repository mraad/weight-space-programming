import argparse
import http.server
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local"))
import sweep
import wsp


def edits(body):
    return f"design text\n\n```edits\n{body}\n```"


BREAKOUT_MAPPING = ("How the 6 channels are built: channel 0 (player) <- paddle; channel 1 (danger) <- nothing; "
                    "channel 2 (reward) <- brick; channel 3 (projectile) <- ball; channel 4 (trail) <- trail; "
                    "channel 5 (gauge) <- nothing. A cell is 1 in a channel if any of its source channels is 1.")


def codex_record(stdout, stderr=""):
    return {"rc": 0, "audit": wsp.audit_events(wsp.parse_codex(stdout, stderr, "gpt-x", 1.0))}


CODEX_START = '{"type":"item.completed","item":{"type":"error","message":"Code Mode is unavailable because code-mode host is disabled."}}\n'
CODEX_REPLY = '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}\n'
CODEX_DONE = '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":7}}\n'


class CodexAudit(unittest.TestCase):
    def test_a_clean_reply_passes(self):
        record = codex_record(CODEX_START + CODEX_REPLY + CODEX_DONE, "Reading prompt from stdin...")
        self.assertIsNone(wsp.run_problem(record))
        self.assertEqual(record["audit"]["answer"], "answer")
        self.assertEqual(record["audit"]["usage"]["output_tokens"], 7)

    def test_a_tool_attempt_seen_only_in_stderr_fails(self):
        record = codex_record(CODEX_REPLY + CODEX_DONE, "ERROR codex_core::tools::router: error=code-mode host is disabled")
        self.assertEqual(wsp.run_problem(record), "the run is not tool-free")

    def test_a_tool_item_in_the_stream_fails(self):
        item = '{"type":"item.completed","item":{"type":"command_execution"}}\n'
        self.assertEqual(wsp.run_problem(codex_record(item + CODEX_REPLY + CODEX_DONE)), "the run is not tool-free")

    def test_a_turn_with_no_end_fails(self):
        self.assertIn("did not finish", wsp.run_problem(codex_record(CODEX_REPLY)))

    def test_a_failed_turn_fails(self):
        failed = '{"type":"turn.failed","error":{"message":"limit"}}\n'
        self.assertEqual(wsp.run_problem(codex_record(CODEX_REPLY + failed)), "the call returned an error")


class EditsLanguage(unittest.TestCase):
    def test_every_form_in_the_prompt(self):
        net, report = wsp.apply_edits(edits(
            "W1[12, 0] = 2.5                     # one entry\n"
            "W1[0:6, 3] = [1, 0, -1, 0, 2, 0]\n"
            "b1[0:4] = -1\n"
            "W1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]\n"
            "W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]\n"
            "W3[:, 2] = 4\n"
            "W3[:, 2] *= 2\n"
            "b3[0] += 3\n"
            "b3[0] -= 1"))
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["applied"], 9)
        self.assertEqual(net["W1"][12, 0], 2.5)
        self.assertEqual(net["W1"][4, 3], 2)
        self.assertEqual(net["W1"][483, 0], 1)
        self.assertEqual(net["W2"][0, 2], 0.5)
        self.assertTrue((net["W3"][:, 2] == 8).all())
        self.assertEqual(net["b3"][0], 2)
        self.assertEqual(net["b1"][:4].tolist(), [-1] * 4)

    def test_list_continues_over_lines_with_comment_and_space_after_bracket(self):
        net, report = wsp.apply_edits(edits("W2[0:2, 0:2] = [ [1, 2],  # first\n  [3, 4] ]"))
        self.assertEqual(report["errors"], [])
        self.assertEqual(net["W2"][1, 1], 4)

    def test_dotted_names_edit_a_copy_of_the_base(self):
        base = {"head.0.weight": np.ones((1, 2, 3, 3), np.float32), "head.0.bias": np.zeros(1, np.float32)}
        net, report = wsp.apply_edits(edits("head.0.weight[0, 0:2, 1, 1] = [5, 6]\nhead.0.bias[0] -= 2\nother.weight[0] = 1"), base=base)
        self.assertEqual((report["applied"], report["statements"]), (2, 3))
        self.assertEqual(net["head.0.weight"][0, :, 1, 1].tolist(), [5, 6])
        self.assertEqual(net["head.0.weight"][0, 0, 0, 0], 1)
        self.assertEqual(net["head.0.bias"][0], -2)
        self.assertEqual(base["head.0.bias"][0], 0)

    def test_only_the_last_block_counts(self):
        net, _ = wsp.apply_edits(edits("b1[0] = 1") + "\n" + edits("b1[1] = 2"))
        self.assertEqual(net["b1"][:2].tolist(), [0, 2])

    def test_bad_statements_are_skipped_and_reported(self):
        net, report = wsp.apply_edits(edits(
            "b1[0] = 1\nb1[0:3] = [1, 2]\nW9[0] = 1\nb1[99] = 1\nb1[1] = 2e7\nb1[2] = [1,\nb3[1] = 5"))
        self.assertEqual(report["applied"], 2)
        self.assertEqual(len(report["errors"]), 5)
        self.assertEqual(net["b1"][0], 1)
        self.assertEqual(net["b3"][1], 5, "an unbalanced line must not swallow the next statement")

    def test_reply_cannot_execute_code(self):
        marker = Path(__file__).with_name("pwned")
        _, report = wsp.apply_edits(edits(f"b1[0] = __import__('pathlib').Path({str(marker)!r}).touch()"))
        self.assertFalse(marker.exists())
        self.assertEqual(report["applied"], 0)

    def test_cut_off_and_missing_blocks(self):
        net, report = wsp.apply_edits("```edits\nb1[0] = 1\nb1[1] = 2")
        self.assertTrue(report["cut_off"])
        self.assertEqual(net["b1"][1], 2)
        _, report = wsp.apply_edits("no block here")
        self.assertFalse(report["block"])
        self.assertEqual(report["nonzero"], 0)


class ReadScoring(unittest.TestCase):
    def test_both_reply_layouts(self):
        plain = wsp.score_read("exp1", "Reasoning: not Pong.\nGame: Frogger, a road crossing\nConfidence: 97%")
        self.assertEqual((plain["guess"], plain["confidence"]), ("freeway", 97))
        heading = wsp.score_read("exp1", "## Game\n**Breakout, a variant**\n\n## Confidence\n**99%**")
        self.assertEqual((heading["guess"], heading["confidence"]), ("breakout", 99))

    def test_unknown_game_and_refusal_are_not_correct(self):
        self.assertEqual(wsp.score_read("exp1", "Game: cannot be determined\nConfidence: 5%")["guess"], "other")
        self.assertEqual(wsp.score_read("exp1", "no fields")["confidence"], None)

    def test_other_wording_does_not_match_the_field(self):
        text = "Games that fit include Pong.\nGame: Pac-Man\n"
        self.assertEqual(wsp.score_read("exp1", text)["guess"], "pacman")


class Control(unittest.TestCase):
    PROMPT = "QUESTION\n\nW1 (layer 1), shape 3 x 2: rows\n" + "\n".join(f"{i} {i}" for i in range(50)) + "\n\nb1 (biases):\n9 9\n"

    def test_shuffle_permutes_w1_rows_only(self):
        out = wsp.shuffle_w1(self.PROMPT)
        before, after = self.PROMPT.split("\n"), out.split("\n")
        self.assertNotEqual(before, after)
        self.assertEqual(sorted(before), sorted(after))
        self.assertEqual(before[:3], after[:3])
        self.assertEqual(before[53:], after[53:])

    def test_shuffle_is_seeded(self):
        self.assertEqual(wsp.shuffle_w1(self.PROMPT, 1), wsp.shuffle_w1(self.PROMPT, 1))
        self.assertNotEqual(wsp.shuffle_w1(self.PROMPT, 1), wsp.shuffle_w1(self.PROMPT, 2))


class Scoring(unittest.TestCase):
    def test_channel_mapping_and_input_order(self):
        channels = {"paddle": 0, "ball": 1, "trail": 2, "brick": 3}
        matrix = wsp.channel_matrix(BREAKOUT_MAPPING, channels)
        state = np.zeros((10, 10, 4), bool)
        state[9, 4, channels["paddle"]] = True
        self.assertEqual(np.flatnonzero(wsp.encode(state, matrix)).tolist(), [(9 * 10 + 4) * 6 + 0])

    def test_channels_merge_several_sources(self):
        line = BREAKOUT_MAPPING.replace("<- paddle;", "<- paddle, ball;")
        matrix = wsp.channel_matrix(line, {"paddle": 0, "ball": 1, "trail": 2, "brick": 3})
        self.assertEqual(matrix[:, 0].tolist(), [1, 1, 0, 0])

    def test_episode_is_reproducible_and_respects_step_cap(self):
        net, _ = wsp.apply_edits(edits("b3[:] = [0, 1, 0, 1, 0, 0]"))
        self.assertEqual(wsp.play(net, "breakout", BREAKOUT_MAPPING, 7), wsp.play(net, "breakout", BREAKOUT_MAPPING, 7))
        self.assertEqual(wsp.play(net, "breakout", BREAKOUT_MAPPING, 7, max_steps=3)[1], 3)


class Lander(unittest.TestCase):
    def test_prompt_holds_the_exact_rules_and_nothing_from_the_scorer(self):
        prompt = wsp.lander_prompt()
        self.assertIn((wsp.LANDER / "sim.js").read_text().strip(), prompt)
        self.assertNotIn("{{", prompt)
        self.assertNotIn("controllerPilot", prompt)
        self.assertNotIn("1000000", prompt, "evaluation seeds must stay hidden")

    def test_network_shapes_and_parameter_count(self):
        net, report = wsp.apply_edits(edits("W1[7, 15] = 1\nW3[15, 3] = 2\nb3[3] = 1\nW1[8, 0] = 1"), wsp.LANDER_SHAPES)
        self.assertEqual(sum(t.size for t in net.values()), 484)
        self.assertEqual(report["applied"], 3, "row 8 of W1 does not exist")

    def test_node_scorer_runs(self):
        result = wsp.lander_eval("random", 30)
        self.assertEqual(result["fixed_target"]["episodes"], 30)
        self.assertEqual(sum(result["fixed_target"]["outcomes"].values()), 30)


class Scrub(unittest.TestCase):
    INIT = {"type": "system", "subtype": "init", "tools": [], "mcp_servers": [], "model": "m", "session_id": "secret-id",
            "cwd": "/private/var/folders/x/T/tmp1", "original_cwd": "/Users/someone/project", "messaging_socket_path": "/tmp/s",
            "plugins": [{"path": "/Users/someone/.claude/plugins/p"}], "uuid": "u1"}
    ASSISTANT = {"type": "assistant", "session_id": "secret-id", "request_id": "r1", "uuid": "u2", "message": {
        "id": "msg_1", "role": "assistant", "model": "m", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "opaque"}, {"type": "text", "text": "hi"}]}}
    RESULT = {"type": "result", "subtype": "success", "result": "hi", "is_error": False, "session_id": "secret-id",
              "credential_expires_at": 123, "total_cost_usd": 0.5, "stop_reason": "end_turn",
              "usage": {"input_tokens": 1, "output_tokens": 2, "inference_geo": "x", "service_tier": "standard"}}

    def test_nothing_personal_survives(self):
        kept = [wsp.scrub_event(e) for e in (self.INIT, self.ASSISTANT, self.RESULT)]
        text = str(kept)
        for secret in ("secret-id", "/private/", "/Users/", "/tmp/", "opaque", "credential_expires_at", "msg_1", "r1", "inference_geo"):
            self.assertNotIn(secret, text)
        self.assertEqual(kept[1]["message"]["content"][0]["thinking"], "plan")

    def test_events_outside_the_allow_list_are_dropped(self):
        self.assertIsNone(wsp.scrub_event({"type": "rate_limit_event", "rate_limit_info": {"plan": "x"}}))
        self.assertIsNone(wsp.scrub_event({"type": "system", "subtype": "thinking_tokens"}))

    def test_a_saved_stream_still_passes_and_fails_the_audit(self):
        clean = [wsp.scrub_event(e) for e in (self.INIT, self.ASSISTANT, self.RESULT)]
        self.assertTrue(wsp.audit_events(clean)["tool_free"])
        use = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}
        self.assertFalse(wsp.audit_events([clean[0], wsp.scrub_event(use), clean[2]])["tool_free"])


INIT_LINE = '{"type": "system", "subtype": "init", "tools": [], "mcp_servers": [], "model": "m", "cwd": "/private/tmp/x"}'
GOOD_LINE = ('{"type": "result", "subtype": "success", "is_error": false, "stop_reason": "end_turn", '
             '"result": "```edits\\nb3[1] = 5\\n```", "usage": {"output_tokens": 9}, "session_id": "secret"}')


class AskAndScore(unittest.TestCase):
    """The whole flow, with a fake `claude` on PATH. Spends nothing."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(wsp, "RESULTS", self.tmp / "results").start()
        mock.patch.dict(os.environ, {"PATH": f"{self.tmp}{os.pathsep}{os.environ['PATH']}"}).start()

    def ask(self, *stdout_lines, timeout=30, exit_code=0, before=""):
        script = self.tmp / "claude"
        lines = "".join(f"printf '%s\\n' '{line}'\n" for line in stdout_lines)
        script.write_text(f"#!/bin/sh\ncat > /dev/null\n{lines}{before}exit {exit_code}\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        args = argparse.Namespace(exp="lander", item="pads", shuffle=False, seed=0, model="m", effort="low", timeout=timeout,
                                  base_url=None, reply_seed=None, extra_body=None)
        try:
            with mock.patch("builtins.print"):
                wsp.ask(args)
            stopped = None
        except SystemExit as err:
            stopped = str(err)
        return next((self.tmp / "results").iterdir()), stopped

    def score(self, run):
        with mock.patch("builtins.print"):
            wsp.score(argparse.Namespace(run=str(run), episodes=6))
        return json.loads((run / "score.json").read_text())

    def test_good_reply_scores_and_saves_nothing_private(self):
        run, stopped = self.ask(INIT_LINE, GOOD_LINE)
        self.assertIsNone(stopped)
        self.assertEqual(self.score(run)["applied"], 1)
        events = (run / "events.jsonl").read_text()
        self.assertNotIn("/private/", events)
        self.assertNotIn("secret", events)

    def test_a_stray_line_does_not_lose_the_reply(self):
        run, stopped = self.ask("warning: update available", INIT_LINE, "[1, 2]", GOOD_LINE)
        self.assertIsNone(stopped)
        self.assertIn("b3[1] = 5", (run / "answer.md").read_text())
        self.assertIn("warning: update available", (run / "stderr.log").read_text())

    def assert_refused(self, *lines, **options):
        run, stopped = self.ask(*lines, **options)
        self.assertIn("run not usable", stopped)
        with self.assertRaises(SystemExit) as refused:
            self.score(run)
        self.assertIn("refusing to score", str(refused.exception))
        self.assertFalse((run / "score.json").exists())
        return run

    def test_an_error_reply_is_refused(self):
        self.assert_refused(INIT_LINE, GOOD_LINE.replace('"is_error": false', '"is_error": true'))

    def test_a_reply_cut_at_the_token_limit_is_refused(self):
        self.assert_refused(INIT_LINE, GOOD_LINE.replace("end_turn", "max_tokens"))

    def test_a_tool_call_is_refused(self):
        use = '{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}'
        self.assert_refused(INIT_LINE, use, GOOD_LINE)

    def test_a_cli_failure_is_refused(self):
        self.assert_refused(INIT_LINE, GOOD_LINE, exit_code=1)

    def test_a_timeout_is_recorded_not_a_crash(self):
        run = self.assert_refused(INIT_LINE, timeout=1, before="sleep 5\n")
        self.assertEqual(json.loads((run / "run.json").read_text())["rc"], 124)
        self.assertIn("timed out", (run / "stderr.log").read_text())

    @unittest.skipUnless(wsp.data_path("exp5", "prompt", "breakout").exists(), "needs wsp.py fetch")
    def test_exp5_rescores_without_a_prompt_in_the_run_folder(self):
        run = self.tmp / "exp5-run"
        run.mkdir()
        (run / "answer.md").write_text("```edits\nb3[0] = 1\n```")
        record = {"exp": "exp5", "item": "breakout", "rc": 0, "audit": {"tool_free": True, "is_error": False, "stop_reason": "end_turn"}}
        (run / "run.json").write_text(json.dumps(record))
        self.assertEqual(self.score(run)["eval"]["episodes"], 6)


class FakeServer(http.server.BaseHTTPRequestHandler):
    reply = {}
    status = 200
    requests = []

    def do_POST(self):
        FakeServer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(FakeServer.status)
        self.end_headers()
        self.wfile.write(json.dumps(FakeServer.reply).encode())

    def log_message(self, *_):
        pass


class LocalServer(unittest.TestCase):
    """The whole flow against a fake OpenAI-compatible server. Needs no GPU."""
    EDITS = "My design.\n\n```edits\nb3[1] = 5\n```"

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(wsp, "RESULTS", self.tmp / "results").start()
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_address[1]}"
        FakeServer.status, FakeServer.requests = 200, []

    def ask(self, finish="stop", content=EDITS, reasoning="long thought", url=None, extra=None, **message):
        FakeServer.reply = {"choices": [{"finish_reason": finish, "message": {"content": content, "reasoning": reasoning, **message}}],
                            "usage": {"prompt_tokens": 7, "completion_tokens": 9}, "id": "secret-id"}
        args = argparse.Namespace(exp="lander", item="pads", shuffle=False, seed=0, model="org/model", effort=None, timeout=10,
                                  base_url=url or self.url, reply_seed=3, extra_body=extra)
        try:
            with mock.patch("builtins.print"):
                wsp.ask(args)
            stopped = None
        except SystemExit as err:
            stopped = str(err)
        return next((self.tmp / "results" / "local").iterdir()), stopped

    def score(self, run):
        with mock.patch("builtins.print"):
            wsp.score(argparse.Namespace(run=str(run), episodes=6))
        return json.loads((run / "score.json").read_text())

    def test_a_good_reply_scores_like_a_claude_reply(self):
        run, stopped = self.ask()
        self.assertIsNone(stopped)
        self.assertEqual(self.score(run)["applied"], 1)
        self.assertEqual((run / "answer.md").read_text(), self.EDITS, "the answer must not include the thinking")
        record = json.loads((run / "run.json").read_text())
        self.assertTrue(record["audit"]["tool_free"])
        self.assertEqual(record["audit"]["usage"], {"input_tokens": 7, "output_tokens": 9})
        self.assertEqual(record["effort"], "default", "a local server has no effort setting to record")

    def test_the_request_offers_no_tools_and_carries_the_seed_and_cap(self):
        self.ask()
        sent = FakeServer.requests[0]
        self.assertNotIn("tools", sent)
        self.assertEqual((sent["seed"], sent["max_tokens"]), (3, wsp.MAX_OUTPUT_TOKENS))
        self.assertEqual(sent["messages"][0]["content"], wsp.SYSTEM_PROMPT)
        self.assertEqual(sent["messages"][1]["content"], wsp.lander_prompt())

    def test_request_options_are_sent_and_recorded(self):
        run, _ = self.ask(extra='{"reasoning_effort": "max", "max_tokens": 200000}')
        self.assertEqual((FakeServer.requests[0]["reasoning_effort"], FakeServer.requests[0]["max_tokens"]), ("max", 200000))
        self.assertIn("200000", json.loads((run / "run.json").read_text())["extra_body"])

    def test_an_option_cannot_offer_tools(self):
        with self.assertRaises(ValueError):
            wsp.call_local("p", self.url, "m", 5, extra={"tools": []})

    def test_a_reply_cut_at_the_cap_is_refused(self):
        run, stopped = self.ask(finish="length")
        self.assertIn("did not finish", stopped)
        with self.assertRaises(SystemExit):
            self.score(run)

    def test_a_tool_call_fails_the_audit(self):
        run, stopped = self.ask(finish="tool_calls", tool_calls=[{"function": {"name": "run_python"}}])
        self.assertIn("not tool-free", stopped)
        self.assertEqual(json.loads((run / "run.json").read_text())["audit"]["tool_calls"], ["run_python"])

    def test_a_server_error_is_refused_with_its_reason(self):
        FakeServer.status = 400
        run, stopped = self.ask(content="context length exceeded")
        self.assertIn("run not usable", stopped)
        self.assertIn("HTTP 400", (run / "stderr.log").read_text())

    def test_a_dead_server_is_refused(self):
        run, stopped = self.ask(url="http://127.0.0.1:9")
        self.assertIn("run not usable", stopped)
        self.assertEqual(json.loads((run / "run.json").read_text())["rc"], 1)

    def test_saved_events_hold_only_what_the_allow_list_keeps(self):
        run, _ = self.ask()
        events = (run / "events.jsonl").read_text()
        self.assertNotIn("secret-id", events)
        self.assertIn("long thought", events)
        self.assertEqual({json.loads(line)["type"] for line in events.splitlines()}, {"system", "assistant", "result"})

    def test_local_runs_stay_out_of_the_folder_the_manual_reads(self):
        self.ask()
        self.assertEqual([p.name for p in (self.tmp / "results").iterdir()], ["local"])


class Sweep(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def run_folder(self, name, seed, stop="end_turn", landed=None, tool_free=True, offered=(), extra=None, answer=None, model="m"):
        run = self.root / name
        run.mkdir()
        audit = {"tool_free": tool_free, "tools_offered": list(offered) if offered is not None else None, "stop_reason": stop, "usage": {"output_tokens": 100}}
        record = {"exp": "lander", "item": "pads", "shuffle": False, "model": model, "reply_seed": seed, "extra_body": extra, "audit": audit}
        (run / "run.json").write_text(json.dumps(record))
        (run / "answer.md").write_text(answer or f"reply {seed}")
        if landed is not None:
            result = {"block": True, "applied": 1, "eval": {"fixed_target": {"landed": landed}, "mid_flight_switch": {"landed": landed / 2}}}
            (run / "score.json").write_text(json.dumps(result))

    def test_a_cut_off_counts_in_the_total_and_a_server_error_does_not(self):
        self.run_folder("a", 0, landed=0.95)
        self.run_folder("b", 1, landed=0.2)
        self.run_folder("c", 2, stop="max_tokens")
        self.run_folder("d", 3, tool_free=False, offered=None, stop=None)
        row = sweep.summarize(sweep.load_runs(self.root))[0]
        self.assertEqual((row["replies"], row["answered"], row["cut_off"], row["server_errors"]), (3, 2, 1, 1))
        self.assertEqual((row["usable"], row["landed_max"], row["landed_switch_max"]), (1, 0.95, 0.475))

    def test_a_restart_asks_again_only_after_a_failed_call(self):
        self.run_folder("a", 0, landed=0.5)
        self.run_folder("b", 1, stop="max_tokens")
        self.run_folder("c", 2, tool_free=False, offered=None, stop=None)
        self.assertEqual(sweep.seeds_left(sweep.load_runs(self.root), "m", "lander-pads", 4), [2, 3])
        self.assertEqual(sweep.seeds_left(sweep.load_runs(self.root), "m", "lander-pads", 4, first=3), [3])

    def test_a_bigger_cap_is_a_separate_condition(self):
        self.run_folder("a", 0, landed=0.5)
        self.run_folder("b", 0, landed=0.9, extra='{"max_tokens": 200000}')
        rows = sweep.summarize(sweep.load_runs(self.root))
        self.assertEqual(len(rows), 2)
        self.assertEqual(sweep.seeds_left(sweep.load_runs(self.root), "m", 'lander-pads {"max_tokens": 200000}', 1), [])

    def test_repeated_replies_are_visible(self):
        self.run_folder("a", 0, landed=0.5, answer="same")
        self.run_folder("b", 1, landed=0.5, answer="same")
        self.assertEqual(sweep.summarize(sweep.load_runs(self.root))[0]["distinct_answers"], 1)

    def test_a_read_task_reports_correct_answers(self):
        run = self.root / "r"
        run.mkdir()
        audit = {"tool_free": True, "tools_offered": [], "stop_reason": "end_turn", "usage": {}}
        (run / "run.json").write_text(json.dumps({"exp": "exp1", "item": "breakout", "shuffle": True, "model": "m", "reply_seed": 0, "audit": audit}))
        (run / "answer.md").write_text("Game: Breakout")
        (run / "score.json").write_text(json.dumps({"guess": "breakout", "correct": True, "confidence": 80}))
        row = sweep.summarize(sweep.load_runs(self.root))[0]
        self.assertEqual((row["task"], row["correct"], row["correct_confidence_median"]), ("exp1-breakout-shuffle", 1, 80))


class Audit(unittest.TestCase):
    INIT = {"type": "system", "subtype": "init", "tools": [], "mcp_servers": []}
    RESULT = {"type": "result", "result": "hi", "is_error": False}

    def test_clean_stream_passes(self):
        self.assertTrue(wsp.audit_events([self.INIT, self.RESULT])["tool_free"])

    def test_tool_use_or_offered_tools_or_no_init_fails(self):
        use = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}
        self.assertFalse(wsp.audit_events([self.INIT, use, self.RESULT])["tool_free"])
        self.assertFalse(wsp.audit_events([{**self.INIT, "tools": ["Read"]}, self.RESULT])["tool_free"])
        self.assertFalse(wsp.audit_events([{**self.INIT, "mcp_servers": [{"name": "x"}]}, self.RESULT])["tool_free"])
        self.assertFalse(wsp.audit_events([self.RESULT])["tool_free"])


if __name__ == "__main__":
    unittest.main()
