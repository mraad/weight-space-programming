# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Tests whether an LLM can read and write neural network weights with no training, after Takehi et al., "Sparks of Weight-Space Programming" (2026). Uses the authors' released networks and prompts, plus Three Pads, an original landing game. Results and method: `docs/index.html`.

## Commands

```sh
uv sync                                      # Python 3.12-3.13; "Cannot import matplotlib" warning is expected
.venv/bin/python wsp.py fetch                # required once: downloads authors' logs into data/

node --test lander/test.js                                          # game and scoring
node --test --test-name-pattern='landing rules' lander/test.js      # single JS test
.venv/bin/python -m unittest discover -s tests                      # parser, scorer, audit
.venv/bin/python -m unittest discover -s tests -k test_cut_off      # single Python test
.venv/bin/playwright install chromium                               # once, for the browser test and screenshots
.venv/bin/python tests/e2e_lander.py                                # browser test; unittest discovery skips it
.venv/bin/python wsp.py verify                                      # no LLM: reproduces authors' scores, writes results/verify.json
```

No build step and no linter. Open `lander/index.html` directly; it works from `file://`, so keep its scripts classic, not ES modules.

These spend LLM capacity, so confirm first:

```sh
.venv/bin/python wsp.py ask lander pads                 # also: ask exp1 breakout [--shuffle], ask exp5 breakout
.venv/bin/python wsp.py score <run>
.venv/bin/python wsp.py lander-export <run>...          # regenerates lander/networks.js for the page
node lander/stress.js <run>/network.json... > results/lander-stress.json
```

Rebuild the manual with `.venv/bin/python docs/capture.py`, then `node docs/build.js`. Check references with `python3 docs/check_refs.py`.

## Architecture

Two languages share one flow. Python (`wsp.py`) calls the LLM and parses replies. Node (`lander/eval.js`) scores Three Pads, because `lander/sim.js` is the single source of the rules for the browser, the scorer and the prompt.

Flow: `lander_prompt()` inserts `sim.js` into `prompt.md` → `ask` writes a run folder under `results/` → `score` parses the last `edits` block into `network.json` and runs the scorer → `lander-export` writes `lander/networks.js` → the page loads it.

MinAtar experiments (`exp5`) score in Python through `play()` and `evaluate()` instead. `channel_matrix()` reads the input mapping from the prompt text, so scoring cannot disagree with what the model saw.

## Changes that must move together

- **Game rules:** `sim.js` is sent verbatim, so any edit changes the prompt and makes saved replies and scores incomparable. Physics values are also typed in `prompt.md` (pad positions, input scaling), `docs/source.html` and the `controllerPilot` tuning in `policy.js`.
- **Network shape:** `LANDER_SHAPES` in `wsp.py`, the network section of `prompt.md`, and the JS test fixtures.
- **Manual:** `docs/index.html` is generated, so edit `docs/source.html`. `build.js` computes tables and figures from `results/`, but `source.html` still has typed numbers in its prose (for example the stress test percentages and "30 of 484"). Recheck those when results change. The build reads `results/verify.json` and `results/lander-stress.json`. It uses every `lander-pads-*` run, so a new run also changes tables and figures.
- **References:** add a citation as `{{cite:id}}` in `source.html` and an entry in `docs/references.json`, then run `python3 docs/check_refs.py`. The build fails on an unknown, unused or failed citation.

## Rules for experiments

- **Public results:** `results/` is public. `scrub_event()` is an allow-list for saved events, so add a field there only if it holds nothing personal or machine-specific.
- **Isolation:** the flags and environment in `call_claude()` and the check in `audit_events()` make a run count as tool-free. `run_problem()` also rejects a failed or cut-off call, and both `ask` and `score` use it, so a tool failure never reads as a bad reply. Do not use `--bare`: it breaks keychain login. Do not run `ask` with tools or from a folder that holds answers.
- **Hidden data:** `policy.js`, its evaluation seeds and the answer keys must stay out of prompts. A test checks the seeds and the plain program. Singh's report saw an agent fit evaluation data once it could read it.
- **Untrusted replies:** model replies are parsed with `ast.literal_eval`. Never `eval` them.
- **Baselines:** keep random and the plain program in comparisons, and do not tune the plain program to lose. Keep unfavorable results, such as the low-fuel failures in `stress.js`.
- **Claims:** write what the sample supports. Most results here are one to three replies.
- **Pinned data:** `SHA` in `wsp.py` fixes the logs commit. `data/` and `results/exp*/prompt.txt` contain GPL-3.0 and CC BY 4.0 material, so do not commit them.
- **Original game:** Three Pads must stay original. Do not copy the name, art or levels of an existing game.

## Documentation style

Manual text follows ASD STE100: short direct sentences, one action per procedure step, one meaning per term (terms are listed in section 1 of the manual). "Flight" means a Three Pads run and "episode" a MinAtar run.
