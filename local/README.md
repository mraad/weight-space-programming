# Local weight changer on gc1

This work tests if a model that runs on gc1 can do the job that GPT-6 Astra did in the paper: read and write network weights with no training.

**Status: paused on 2026-11-06, in Step 3 of 3. Nothing runs on gc1. Go to [Restart here](#restart-here) to continue.** This file records only what was measured. It marks each estimate as an estimate.

## Restart here

First, check gc1. A stop of the instance erases the NVMe disk. If `/opt/dlami/nvme/wsp` is empty, copy the repo, then run `local/gc1-setup.sh`.

Then continue in this order:

| Order | Work | Note |
|---|---|---|
| 1 | Finish Three Pads at the 200,000-token cap | 9 replies are not done: seeds 8 and 12 to 19. Use the command below. The sweep skips the 11 finished seeds. |
| 2 | Qwen Breakout write at the 200,000-token cap, 12 replies | This is the deciding test. Estimate: about 2.5 hours for each round of 12. |
| 3 | Qwen Breakout read and its shuffled control | Do this if the write test shows skill. |

```sh
bash local/serve.sh qwen                # wait until ports 8001 to 8004 are healthy
export PATH="$HOME/.local/bin:$PATH"
.venv/bin/python local/sweep.py run --model Qwen/Qwen3.8-27B \
  --urls http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004 \
  --task lander-pads --replies 20 --per-server 3 --timeout 21600 --extra-body '{"max_tokens": 200000}'
```

Change `--task` and `--replies` for the Breakout write test. Run all of these inside tmux session `wsp`.

Rule for the Breakout write test, set before the result is known:

| Qwen's best mean score | Meaning | Next |
|---|---|---|
| Below 25.3 (PPO at 500M steps) | No real skill at this size | Stop, and write the result |
| Above 25.3, far below Claude's 269.4 | Some real skill | Discuss tuning for this gap |
| Near Claude | A strong result. It is not expected | Check it hard, then go to Step 6 |

## Questions

1. **Size:** can a stock local model do this task, and does a bigger model do better?
2. **Tuning:** does tuning add a skill that works on tasks the model never saw, or does the model only memorize a template?

The authors do not say that Astra was tuned for this. They credit its reasoning.

## Plan

| Step | Work | State |
|---|---|---|
| 1 | Harness: `wsp.py ask --base-url`, `local/sweep.py` | Done |
| 2 | Set up gc1, measure fit and speed | Done for Qwen |
| 3 | Baseline sweep with no tuning, 20 replies per model per task | Paused. Nemotron 3 Super is done. Not done for Qwen: Three Pads at the 200,000-token cap (11 of 20), Breakout write at that cap, and Breakout read with its control |
| 4 | Proposal: tune Qwen with a compiler as the teacher | Needs approval |
| 5 | Proposal: tune with scorer-selected replies | Needs approval |
| 6 | Proposal: apply the changer to another local model | Needs approval |

### Models

| Model | Params | License | Use |
|---|---|---|---|
| Qwen/Qwen3.8-27B | 27.8B | Apache-2.0 | The weight changer. gc1 can run it and tune it. |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 123.6B | NVIDIA Nemotron Open Model License | The bigger reference. One copy over all 4 cards. |

Nemotron 3 Ultra (560B, NVFP4) was tried first and did not start. See Problems found.

**DeepSeek models are not allowed in this project. Do not download, serve or evaluate them.**

### Rules for tuning data

- Do not train on Claude or Astra replies. The vendors' terms limit the use of output to train other models.
- Keep Three Pads, `lander/policy.js` and the evaluation seeds out of the training data. Three Pads is the held-out test.

## gc1

Measured on `g7e.24xlarge`: 4 RTX PRO 6000 Blackwell cards with 96 GB each, no NVLink, 1 TB RAM.

All work is in `/opt/dlami/nvme/wsp`, and all remote jobs run in tmux session `wsp`. Each server has its own window and its own log in `/opt/dlami/nvme/wsp/logs`.

**The NVMe disk is ephemeral. A stop of the instance erases the repo copy, the venvs, the models and the results.** Copy `results/local/` to the Mac after each sweep.

### Commands

Run these on gc1, from `/opt/dlami/nvme/wsp/weight-space-programming`. Run each one inside tmux session `wsp`.

```sh
bash local/gc1-setup.sh                 # once: vLLM, ninja, repo data, models
bash local/serve.sh qwen                # one server per card, ports 8001 to 8004
bash local/serve.sh nemotron            # one server over all 4 cards, port 8001; download the model first
bash local/serve.sh status              # GPU memory, GPU load, server health
bash local/serve.sh stop

export PATH="$HOME/.local/bin:$PATH"    # the scorer needs node
.venv/bin/python local/sweep.py run --model Qwen/Qwen3.8-27B \
  --urls http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004 \
  --task lander-pads --replies 20
.venv/bin/python local/sweep.py summary
```

A sweep can restart. It skips a reply seed that has an answer or a cut-off, and it asks again after a failed call.

Local runs go to `results/local/`. That folder keeps them out of the manual, because `docs/build.js` reads only `results/lander-pads-*`. It also keeps the licensed Exp 1 and Exp 5 prompts out of git, through `.gitignore`.

## Measurements

Qwen3.8-27B on vLLM 0.33.0:

| Item | Value | Note |
|---|---|---|
| Memory used per card | 88,359 MiB of 97,887 MiB | Reserved at startup |
| Context cache per card | 543,646 tokens | |
| Replies at the full 262,144-token context | 2.07 per card | vLLM's figure |
| Generation speed, one reply | 26.5 tokens per second | One 10-second sample, not a benchmark |
| Generation speed, 17 replies on 4 cards | About 430 tokens per second | One early sample. Each reply kept about 25 tokens per second |

Nemotron 3 Super on vLLM 0.33.0:

| Item | Value | Note |
|---|---|---|
| Weights per card | 57.01 GiB | Equal on all 4 cards |
| Memory used per card | About 89,500 MiB of 97,887 MiB | Reserved at startup |
| Context cache | 10,708,084 tokens | 40.85 replies at the full 262,144-token context |
| Generation speed, one reply | About 109 tokens per second | All 4 cards at 99% |
| Generation speed, 8 replies | Median 392 tokens per second | 27 log samples. About 49 tokens per second per reply |

The scorer agrees across machines: a saved Claude network scored on gc1 matches its saved `score.json` exactly.

### Results for all tasks, no tuning

All rows use the same prompts as the Claude runs, a 64,000-token cap and no tools.

| Task, 20 replies | Measure | Qwen3.8-27B | Nemotron 3 Super | Reference |
|---|---|---|---|---|
| Three Pads | Usable replies | 4 | 0 | Claude 3 of 3 |
| Three Pads | Best landing rate, fixed-pad test | 100% | 6.3% | Plain program 99.8% |
| Three Pads | Cut off | 14 | 0 | |
| Three Pads | Median output tokens | 64,000 | 7,409 | Claude 20,617 to 29,907 |
| Breakout write | Answered | 1 | 20 | |
| Breakout write | Best mean score | 7.957 | 0.576 | PPO 25.3, Claude 269.4, Astra 517.4 |
| Breakout read | Correct | Not run | 0 of 13 answered, 7 cut off | Claude correct |
| Breakout read, shuffled | Correct | Not run | 2 of 18 answered, 2 cut off | Claude correct |

A Nemotron reply with no statements scored 0.508 on Breakout write. Thus, its best score of 0.576 does not show skill.

The usable counts on Three Pads, 4 of 20 and 0 of 20, do not separate the models by themselves: a two-sided Fisher exact test gives p of about 0.11, and the Wilson intervals overlap (Qwen 8% to 42%, Nemotron 0% to 16%). The landing rates show a larger difference, but they are from a selected best reply.

The two models use their tokens differently. Most Qwen replies reach the cap. Nemotron stops early on Three Pads and Breakout write. This sample does not show the cause.

### Three Pads, Qwen3.8-27B, 200,000-token cap

This run stopped on request, with 11 of 20 replies finished. No reply was cut off. The 11 replies are the shortest ones, because the longer replies had not finished. Thus, 4 of 11 is not a usable rate for the sweep.

| Seed | Output tokens | Fixed-pad test | Pad change |
|---|---|---|---|
| 7 | 84,532 | 100% | 96.0% |
| 2 | 51,054 | 99.9% | 92.1% |
| 0 | 64,542 | 99.5% | 81.6% |
| 9 | 74,716 | 98.3% | 66.9% |
| 6 | 79,094 | 87.6% | 92.5% |
| 3 | 49,857 | 85.6% | 66.2% |
| 11 | 68,348 | 78.7% | 44.0% |
| 1 | 61,972 | 60.7% | 35.1% |
| 4 | 71,774 | 51.5% | 16.0% |
| 10 | 84,790 | 3.7% | 2.9% |
| 5 | 61,796 | 3.0% | 2.7% |

- 4 replies are usable. 3 of the 4 used more than 64,000 tokens, so the old cap would stop them.
- The final count for 20 replies will be 4 to 13. The Gate 1 bar is 10, so it needs 6 of the 9 replies that are not done.
- A long reply does not always succeed. Seeds 5 and 10 used 61,796 and 84,790 tokens and landed under 4%.
- The same seed does not give the same reply. Seed 3 landed 100% in the 64,000-token run and 85.6% here. Seed 2 landed 66.7% and 99.9%. The cause is not tested.

**Gate 1, so far:** no model reaches usable replies in half of its tries. The bigger model is not better. Qwen is not near random, so the gate does not point directly to tuning. The token cap is a real limit for Qwen, but the result at the larger cap is not complete.

**Verdict, so far:** a stock local model can write working weights for Three Pads in some tries. There is no sign yet that one can do the Astra-size job. Three Pads may be too easy to show this, and Qwen is barely tested on Breakout.

## Thought experiment: how to tune a weight changer

This section is a design. No part of it has been run. The numbers in it come from the runs above, or they are marked as a choice.

### The skill to train

The model must read a task and a network shape, and then write weights that score well, with no tools and no training run. The data above shows three separate gaps:

| Gap | Evidence | What to train |
|---|---|---|
| Cost | Qwen uses 49,857 to 84,790 tokens on 484 weights | Shorter derivations that still reach the answer |
| Checking | Seeds 5 and 10 think for a long time and land under 4% | Trace the network by hand on sample inputs, and find the error before the answer |
| Scale | Breakout has 20,486 weights, and a reply cannot list them all | Sparse designs and slice statements, such as `b2[0:16] = 0` |

### Training data

| Source | Example | Why it is useful | Limit |
|---|---|---|---|
| Forward-pass drills | Given weights and an input, compute the output | The answer is exact and free, with no limit on the amount | It teaches reading, not design |
| Compiler teacher | A program builds a controller for a generated task, and writes the derivation and the edits | It gives correct worked examples | The model can learn the compiler's one template |
| Scorer as reward | Sample replies on training tasks, and reinforce the ones that score well | It needs no outside teacher and rewards the true goal | It can find faults in the simulator. It is also the most expensive |
| Repair | Damage a working network, and ask for the fix | It is closer to Exp 4 and needs few new parts | It is a smaller skill than writing from nothing |

Order of work: drills, then small compiled tasks, then larger tasks, then the scorer as reward. Generate tasks with changing physics, input order, activation functions and network shapes, so that one template cannot solve them all.

### Criteria for an example

An example enters the training set only if all of these are true:

- Its network passes in its own simulator and beats the random network and the plain program for that task.
- A program recomputes each number in the derivation, and each number agrees.
- It contains no Claude or Astra output.
- It contains no Three Pads rules, `policy.js` or evaluation seeds. A test scans the file for them.

### Criteria while tuning

Watch these on a validation set of task families that training did not use. Stop when the score stops rising, or when the training score rises and the validation score does not.

| Measure | What it shows |
|---|---|
| Share of replies that parse | The format is learned |
| Share of replies that finish inside the cap | The cost gap is closing |
| Median score against the plain program | The design skill, in a fair unit |
| Number of distinct designs across tasks | Low means a template. Compare the weight patterns between tasks |

Use the real score as the reward. Give no reward for reply length, and no credit for a reply that does not parse.

### Criteria for success

Test at 50 replies per tier, with the same prompts and the same token cap as the stock run:

| Tier | Task | A pass shows |
|---|---|---|
| T0 | New parameters, same task family | Training worked. It proves nothing more |
| T1 | Three Pads, never seen | Near transfer |
| T2 | Breakout write, never seen | Far transfer |
| Control | Breakout read with shuffled W1, and the Three Pads stress test | The model reads the weights, and the design holds outside the normal starts |

- **Success:** the usable rate on T1 beats the stock rate, with no overlap in the 95% Wilson intervals. The stock rate must come from a complete sweep at the same cap.
- **A template, not a skill:** T0 is high and T1 stays low. Record this as the result, and stop.
- **The main goal:** T2 beats PPO's 25.3. Only this result moves toward the Astra-size job.
- **No extra rounds to improve a held-out score.** After the test uses T1 or T2 to select a model, that tier is no longer held out, and a new one is necessary.

The thresholds of 50 replies and a Wilson test are choices in this design, not measured values.

### Three Pads, Qwen3.8-27B, no tuning

20 replies, 64,000-token cap, no tools. The prompt is identical to the Claude prompt.

| Outcome | Replies |
|---|---|
| Usable: lands at least 90% on the fixed-pad test | 4 |
| Answered, not usable (66.7% and 0%) | 2 |
| Cut off at the cap, no answer | 14 |

The usable rate is 20%, with a 95% Wilson interval of about 8% to 42%. The 4 usable replies landed 96.3% to 100% on the fixed-pad test and 85.0% to 96.8% with a pad change. Replies that finished used 49,537 to 62,996 tokens. Claude used 20,617 to 29,907 tokens on this task.

The stress test ran on one usable reply (seed 3), with Claude reply 12064 and the plain program beside it:

| Start | Qwen seed 3 | Claude | Plain program |
|---|---|---|---|
| Normal, fresh seeds | 100% | 100% | 99.8% |
| Falling at vy -1.0 | 39.9% | 99.1% | 98.4% |
| Low, y = 15 | 28.6% | 100% | 99.9% |
| Low, y = 8 | 17.0% | 100% | 100% |
| Fuel 150 | 2.8% | 59.0% | 43.4% |
| Sideways speed up to +-1.0 | 63.2% | 68.8% | 69.2% |

Three Pads may be too easy to separate strong models from weak ones, because the plain program also scores 99.8%.

### How to read GPU memory and GPU load

Full memory with low load is normal.

- **Memory** shows what vLLM reserved at startup, not what the model uses now. `--gpu-memory-utilization 0.92` makes each server take about 88 GB, even with no requests.
- **Load** shows the current work. One reply uses one card. A card with no reply stays near 0%.

During the live check, card 0 showed 100% and 399 W, and the other three cards showed 0%.

## Problems found

| Symptom | Cause | Fix |
|---|---|---|
| vLLM stopped at startup: `No such file or directory: 'ninja'` | FlashInfer compiles its kernels at startup. `ninja` was not installed. | `gc1-setup.sh` installs `ninja`. `serve.sh` adds `ninja` and `nvcc` to `PATH`. |
| `flashinfer-cubin==0.7.2.post1` did not install | The prebuilt package has no version that matches FlashInfer 0.7. | Do not use it. Compile the kernels. |
| Setup printed `SETUP-DONE` after a failed step | The script continued after an error. | `gc1-setup.sh` now uses `set -e`. |
| 4 servers could compile the same kernels at the same time | All servers shared one kernel cache. This risk was not observed as a failure. | `serve.sh` waits for the first server before it starts the others. |
| The scorer failed under tmux | `node` is in `~/.local/bin`, which is not on `PATH` for a tmux job started over SSH. | Use `export PATH` in the same job, as the commands above show. `tmux set-environment` did not reach new windows here. `sweep.py` now stops before it asks if `node` is missing. |
| `serve.sh stop` ended the tmux session | The command closed the last window. | Keep another window open, or start a new session. |
| Nemotron 3 Ultra stopped at the startup memory test: `Tried to allocate 9.87 GiB`, 5.10 GiB free | The weights take 85 GiB of each 94.97 GiB card, and the card's batch of 32,768 tokens needs 9.87 GiB more. | A batch of 4,096 tokens passed this test and needed 1.24 GiB. |
| Nemotron 3 Ultra stopped in the FlashInfer warm-up at 97% memory: `Tried to allocate 1.24 GiB`, 1.21 GiB free | The context cache took 4.33 GiB per card before the warm-up ran. The other two workers then hung at 100% load. | Not proven. A start at 95% memory did not run, because of a typing error in the flag. Use of this model stopped at the user's request. |

## Limits so far

- Each model has one set of default settings. The temperature is each model's own default.
- The size comparison is one pair of models with different architectures. Nemotron 3 Ultra did not run.
- The price of `g7e.24xlarge` time is not recorded here.
- The 64,000-token output cap matches the Claude runs. The vendors recommend larger limits, so a model can fail here only because of the cap. A run with a larger cap must appear as a separate row in the summary, and `--extra-body` does this.
