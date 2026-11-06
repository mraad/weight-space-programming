# Initial project: replicate Weight-Space Programming on Claude

Goal: test the claims of Takehi et al. (Nov 2026) on another frontier model, using their released
networks and prompts, and add the negative control the post lacks. No training.

Source data: github.com/RikiyaT/lm_understand-logs, pinned at `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
Logs and networks are CC BY 4.0. Prompts for Exp 5 quote GPL-3.0 MinAtar source, so `data/` is fetched
at runtime and never committed.

## Plan

- [x] `wsp.py fetch`: download Exp 1, 2 and 5 files at the pinned commit into `data/`.
- [x] `wsp.py ask`: send one prompt to `claude -p` with no tools, no settings, no MCP, from an empty
      working directory so it cannot read answer keys. Save the raw event stream and fail the run if
      any tool call appears (same audit as the authors).
- [x] `wsp.py score`: map the `Game:` / `Images:` line to an answer by keywords, as the authors do.
- [x] `wsp.py control`: rebuild the Exp 1 prompt with weights destroyed (W1 rows shuffled, or random
      init at the same scale). A reader that still names a game with high confidence is confabulating.
- [x] `wsp.py apply`: parse the `edits` language into the 600-32-32-6 network.
- [x] `wsp.py eval`: score with MinAtar 1.0.15 under the post's protocol (1,000 episodes, sampled
      actions, sticky 0.1, 10,000 step cap).
- [x] Offline proof, no LLM: parsing Astra's `answer.md` gives exactly `hand.npz`, and evaluating
      `hand.npz` lands within error of 517.4 (Breakout), 53.1 (Seaquest), 53.2 (Asterix).
- [x] Unit tests for the parser and the scorer. README with commands and results.
- [x] Live trial: Exp 1 Breakout, its control, and Exp 5 Breakout on Claude. Save raw output and
      report results as they fall, including failures.

- [x] Three Pads: original game in `lander/`, three live replies, stress test, page.
- [x] Manual in `docs/index.html`.

Plan changes: `control`, `apply` and `eval` became `ask --shuffle` and `score`.

## Later, not in this version

True null control for Exp 1 (random weights, no weights). All Exp 1 networks with replicates. Heuristic-program baseline for Exp 5. Harder Three Pads with a PPO baseline. Exp 2 and 3 live runs, Codex backend, Exp 4, 6, 7, 8.

# Tank repair by weight edits (branch tank-weight-changer)

Plan and gates: `geo/README.md`. The changer is Claude Opus 5.5 only.

- [x] Parser: dotted tensor names, edits start from a base network. Existing tests and `verify` unchanged.
- [x] Stage A: harness, baselines, Gate 0 on gc1. The fault exists.
- [x] No-LLM check of the scoring path: the LDA edits score the same as the LDA baseline.
- [x] User confirms the LLM spend.
- [x] Stage B: `ask tanks repair` x 3, `ask tanks repair-shuffled` x 3, score on gc1, Gate 1. Result: no skill, control did not collapse.
- [x] Rounds: not needed, the replies beat no edit.
- [x] Stage C: skipped by Gate 1, then run at the user's request under its own rule. Result: some design skill, no practical gain.
- [x] Stage C with GPT-6 Astra through Codex, same prompt and rule. Result: the same verdict.
- [x] User decides: close it.
- [x] `final`: open the test cells once. Write the result.
