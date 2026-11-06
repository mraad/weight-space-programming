# Tank repair by weight edits

This work tests if an LLM can repair a detection network by editing its weights, with no training.

**Status: closed. Gate 1 reads "no skill", so Stage C did not run. The test cells were opened once, and the result is below.**

**Result: use a plain fix or retrain. On the test cells, the LLM repair is better than no edit, but it does not show practical gain over histogram match or AdaBN.**

**Stage C, run after that at the user's request: Claude and GPT-6 Astra each wrote working tank detectors from zeros, with no training. Under the rule set before the replies, both results are "some design skill, no practical gain".**

## Final result

Test cells of `Tanks.tif`: 28 tanks in 38 cells. These cells did not select any network or reply. Intervals are 95%, from 2,000 resamples of the cells. "Against no edit" is the interval of the difference on the same resamples.

| Comparator | Pixel IoU | 95% interval | Object F1 | 95% interval | Found, false, missed | IoU against no edit | F1 against no edit |
|---|---:|---|---:|---|---|---|---|
| No edit, seed 0 | 0.349 | 0.214 to 0.454 | 0.174 | 0.058 to 0.356 | 6, 35, 22 | | |
| No edit, seed 1 | 0.294 | 0.128 to 0.473 | 0.179 | 0.049 to 0.323 | 6, 33, 22 | -0.115 to +0.037 | -0.164 to +0.146 |
| No edit, seed 2 | 0.242 | 0.097 to 0.355 | 0.000 | 0 to 0 | 0, 8, 28 | -0.198 to -0.005 | -0.356 to -0.058 |
| LDA head | 0.475 | 0.321 to 0.596 | 0.222 | 0.075 to 0.458 | 6, 20, 22 | +0.056 to +0.217 | -0.013 to +0.180 |
| Head retrained by gradient | 0.487 | 0.339 to 0.606 | 0.207 | 0.070 to 0.426 | 6, 24, 22 | +0.070 to +0.231 | -0.025 to +0.152 |
| Filter LDA, no network | 0.524 | 0.325 to 0.677 | 0.245 | 0.082 to 0.516 | 6, 15, 22 | +0.058 to +0.292 | -0.057 to +0.267 |
| **Claude repair** | **0.598** | 0.460 to 0.711 | **0.449** | 0.303 to 0.606 | 11, 10, 17 | +0.160 to +0.349 | +0.095 to +0.417 |
| Claude control 41910 | 0.558 | 0.435 to 0.652 | 0.423 | 0.250 to 0.688 | 11, 13, 17 | +0.141 to +0.302 | +0.057 to +0.444 |
| Claude control 41914 | 0.080 | 0.040 to 0.128 | 0.013 | 0.005 to 0.024 | 5, 731, 23 | -0.366 to -0.160 | -0.341 to -0.045 |
| Histogram match | 0.588 | 0.443 to 0.696 | 0.600 | 0.261 to 0.836 | 15, 7, 13 | +0.116 to +0.374 | +0.067 to +0.729 |
| AdaBN | 0.604 | 0.464 to 0.707 | 0.591 | 0.361 to 0.839 | 13, 3, 15 | +0.159 to +0.355 | +0.150 to +0.655 |
| Colour jitter in the 2015 training | 0.660 | 0.549 to 0.757 | 0.431 | 0.250 to 0.694 | 11, 12, 17 | +0.228 to +0.411 | +0.050 to +0.467 |
| Network trained on `Tanks.tif` (upper bound) | 0.806 | 0.751 to 0.844 | 0.793 | 0.577 to 1.000 | 21, 4, 7 | +0.355 to +0.573 | +0.342 to +0.843 |

What the test cells show:

- The repair is better than no edit. Both intervals of the difference are above 0.
- The repair and the plain fixes have overlapping intervals. Its IoU is near theirs, and its point F1 is lower. The run has no direct test of repair against each plain fix.
- The repair and control 41910 have overlapping intervals. The test cells agree with Gate 1: this control does not separate a reader from a copier.
- Control 41914 made 731 false objects.
- Training on the new image is the only row with an interval above the repair on IoU. Its interval starts at 0.751, and the repair's ends at 0.711.

## The fault

A ResNet18 U-Net (14,328,209 weights) trains on an image from 2015. It then runs on `Tanks.tif`, an image of the same area from another date. Both images are on one pixel grid, so scale is not part of the fault.

`finetune/data/split_cells.geojson` divides the area into cells: 153 train, 30 val, 38 test. Pixels in other partitions are hidden during each fit, and evidence uses train cells only.

## Stage A results

Val cells of `Tanks.tif`: 19 tanks. Object F1 uses IoU >= 0.5. Each row is one network, so differences are not yet tested. Test cells will give intervals.

| Comparator | What it uses | Pixel IoU | Object F1 |
|---|---|---:|---:|
| No edit (seed 0, the repair target) | Nothing | 0.386 | 0.140 |
| No edit (seed 1) | Nothing | 0.257 | 0.118 |
| No edit (seed 2) | Nothing | 0.350 | 0.148 |
| LDA head, from the evidence table | Train labels | 0.485 | 0.128 |
| Head retrained by gradient | Train labels | 0.493 | 0.128 |
| AdaBN | New pixels, no labels | 0.481 | 0.578 |
| Histogram match | New pixels, no labels | 0.578 | 0.500 |
| Filter LDA, no network | Train labels | 0.514 | 0.286 |
| Colour jitter in the 2015 training | No new data | 0.701 | 0.649 |
| Network trained on `Tanks.tif` (upper bound) | Train labels, val selects the epoch | 0.837 | 0.842 |

**Gate 0 passes.** No edit is 46% of the upper bound. The rule was to stop at 90%.

What this shows, before any LLM call:

- A head edit has a low limit. Real gradients on the head reach only 0.493 IoU and 0.128 object F1. A reply must change more than the head to do better.
- Plain fixes are strong. Histogram match and AdaBN need no labels and no LLM. A reply must beat them to have practical use.
- Colour jitter prevents most of the fault at training time. If you can retrain, use it.

## Stage C result

Claude Opus 5.5 at `xhigh`, no tools, one reply each. All 3 replies finished, passed the tool audit and parsed with no errors. They used 61,264 to 84,962 output tokens, so each was near or above the 64,000 mark. They give 3 different designs. Each one finds pale regions at more than one scale.

| Network | Nonzero weights | Val IoU | Val object F1 | Test IoU (95%) | Test object F1 (95%) | Test found, false, missed |
|---|---:|---:|---:|---|---|---|
| Fact rule, gray midpoint | 7 | 0.131 | 0.038 | 0.143 (0.075 to 0.215) | 0.049 (0.019 to 0.088) | |
| Filter LDA, fitted on train pixels | | 0.514 | 0.286 | 0.524 (0.325 to 0.677) | 0.245 (0.082 to 0.516) | 6, 15, 22 |
| **Claude 54466** | 409 | 0.479 | 0.294 | 0.517 (0.324 to 0.670) | 0.286 (0.139 to 0.500) | 7, 14, 21 |
| **Claude 54464** | 140 | 0.505 | 0.217 | 0.556 (0.400 to 0.690) | 0.295 (0.160 to 0.471) | 9, 24, 19 |
| **Claude 54468** | 305 | 0.543 | 0.231 | 0.576 (0.422 to 0.701) | 0.278 (0.179 to 0.400) | 10, 34, 18 |
| Same network, trained on `Tanks.tif` | All | 0.710 | 0.552 | 0.652 (0.475 to 0.772) | 0.455 (0.258 to 0.720) | 10, 6, 18 |

Zero and random networks score 0.000 to 0.028 val IoU, and object F1 0.

**The rule reads "some design skill, no practical gain".** The median val IoU is 0.505. It is above the fact rule (0.131) and below filter LDA (0.514). The difference from filter LDA is small, and each reply is close to it, so this is a near tie and not a clear loss. On the test cells, the median reply is above filter LDA on IoU (0.556, 0.524) and object F1 (0.286, 0.245), but the intervals overlap.

These plain colour rules use only the fact sheet. They came after the replies, because the fact rule was a weak bar, so they do not change the verdict:

| Plain rule from the fact sheet | Val IoU | Val object F1 | Test IoU | Test object F1 |
|---|---:|---:|---:|---:|
| Band LDA, midpoint | 0.145 | 0.039 | 0.156 | 0.051 |
| Band LDA, midpoint and tank share | 0.385 | 0.082 | 0.427 | 0.183 |
| Index 1.5B - 0.5R from replies 54464 and 54468, midpoint | 0.301 | 0.091 | 0.335 | 0.118 |

All 3 replies beat all of these rules on val and test IoU and on object F1. Thus the multi-scale design adds value past the colour numbers in the table. The sample is 3 replies on one image.

Limits:

- The prompt names the task and says that tanks are pale discs. The fact sheet comes from labels on the train cells, so "no training" does not mean "no labels".
- The object scores are low. The replies miss 18 to 21 of 28 test tanks, and they make 14 to 34 false objects. They find pale regions and do not test for a circle, as reply 54464 says.
- A trained copy of the same network does better on val. Its test interval overlaps the replies. Its training time was not measured.
- The test cells were opened before this stage, so they only report. No reply was selected on them.
- Each reply used about 1 to 1.7 dollars of Claude capacity.

## Stage C, GPT-6 Astra

The same prompt and the same rule. GPT-6 Astra at `xhigh`, through `codex exec`, one reply each. All 3 replies finished, parsed with no errors, and made no tool attempts. They used 22,358 to 28,886 output tokens and took 17 to 20 minutes each.

| Network | Nonzero weights | Val IoU | Val object F1 | Test IoU (95%) | Test object F1 (95%) | Test found, false, missed |
|---|---:|---:|---:|---|---|---|
| Astra 60743 | 345 | 0.219 | 0.036 | 0.225 (0.117 to 0.324) | 0.066 (0.031 to 0.106) | 17, 472, 11 |
| Astra 60745 | 507 | 0.460 | 0.171 | 0.516 (0.381 to 0.605) | 0.258 (0.134 to 0.364) | 16, 80, 12 |
| Astra 60748 | 525 | 0.501 | 0.214 | 0.551 (0.414 to 0.659) | 0.316 (0.158 to 0.435) | 15, 52, 13 |
| Median, Astra | | 0.460 | 0.171 | 0.516 | 0.258 | |
| Median, Claude | | 0.505 | 0.231 | 0.556 | 0.286 | |

**The rule reads "some design skill, no practical gain", the same as Claude.** The median val IoU is 0.460. It is above the fact rule (0.131) and below filter LDA (0.514).

What the two samples show:

- The two models make different errors. Astra finds more test tanks (15 to 17, against 7 to 10), and it makes more false objects (52 to 472, against 14 to 34).
- The Astra replies test for shape, with rings or opposing directions at several radii. The Claude replies mainly fill pale regions. This can explain the higher recall, but this test does not prove it.
- One Astra reply fails on false objects (472). No Claude reply fails in this way.
- This does not show that one model is better. The sample is 3 replies for each model, and the intervals overlap.
- Astra's Breakout lead in the paper did not appear here. This is one task.

Limits of this comparison:

- Codex has its own base instructions. The Claude runs use a short system prompt.
- Codex still names some tools to the model. No flag removes them. A probe showed that `exec` and `spawn_agent` fail closed. The audit rejects a run if an attempt appears in the stream or in the Codex error log. There were none.
- The token counts may not be comparable between the two tools.
- OpenAI received the fact sheet numbers and no pixels, with the user's approval.

## Stage C rule, set before any reply

The user asked for Stage C after Gate 1 read "no skill", so this run is outside that gate. Claude writes all weights of `geo/tiny.py` (16,961 weights) from zeros, with no tools. The prompt holds the source and a fact sheet from train cells. It has no per-pixel data and no answer table.

The result is the median of 3 replies on val cells. All 3 replies are reported. The test cells were opened by `final`, so they only report here and select nothing.

| Median of 3 replies, val IoU | Meaning |
|---|---|
| At or below the fact rule | No skill past the fact sheet. A plain program does this from the same numbers |
| Above the fact rule, at or below filter LDA | Some design skill, no practical gain |
| Above filter LDA | A strong result. It is not expected. Check it hard on a new scene |

## Stage B results

Claude Opus 5.5 at `xhigh`, no tools, one reply each. All 6 replies finished and passed the tool audit. Val cells, 19 tanks.

| Reply | Edit | Weights changed | Pixel IoU | Object F1 |
|---|---|---:|---:|---:|
| Repair, 3 replies with the same effect | Copy the new `encoder.bn1` statistics, all 64 channels | 128 (0.0009%) | 0.553 | 0.449 |
| Control 41910 and 41912, same effect | Copy the shuffled rows, and skip rows that look dead | 112 (0.0008%) | 0.613 | 0.400 |
| Control 41914 | Copy the shuffled rows, all 64 | 128 (0.0009%) | 0.093 | 0.016 |

**Gate 1 reads "no skill".** The median repair beats no edit (0.386, 0.140), but the median control (0.613, 0.400) does not collapse. Stage C does not run. Feedback rounds do not run, because the replies beat no edit.

The repair is in the range of the plain fixes. Its IoU is above AdaBN (0.481) and below histogram match (0.578). Its object F1 is below both (0.578, 0.500).

### Why the control did not collapse

These checks came after the result, so they explain it and do not change the verdict. They use plain programs, with no LLM.

| `encoder.bn1` edit, no LLM | Val IoU | Val object F1 |
|---|---:|---:|
| True statistics, all 64 channels (same as the repair) | 0.554 | 0.449 |
| True variance only | 0.508 | 0.390 |
| True mean only | 0.101 | 0.088 |
| One scale on all live variances (x 7.82), no channel data | 0.500 | 0.308 |
| 5 random shuffles, skip rows with near-zero variance | 0.244 to 0.681 | 0.085 to 0.667 |
| The same 5 shuffles, write all rows | 0.043 to 0.310 | 0.000 to 0.056 |

- Most of the repair is one change: larger stored variances. Thus the row order has a small effect, and this control cannot separate a real reader from a copier.
- A shuffle that writes a near-zero variance on a live channel fails each time. That explains control 41914.
- The range of the random shuffles is wide, and it includes the repair. With 19 tanks, a difference between these rows is weak evidence.

### What the test does not show

- **The prompt gave away the answer.** The evidence table lists the new `encoder.bn1` statistics, channel by channel. The replies found the right layer and copied the values. A program that picks the layer with the largest shift gets the same network.
- **The 3 repairs are one result.** They have the same effect, so they are not 3 independent tests of skill.

### An observation outside the gate

All 3 control replies found the shuffle with no prompt to look for it. Each one multiplied kernel sums by channel means, and found that the prediction did not match the measured logits (about -12.8 against +1.29 on tank interior). Each one then left the head alone. No repair reply reported a shuffle. This is 3 of 3 against 0 of 3, and it was not a planned test. It suggests that the replies read the table against the weights. The replies did not check the `encoder.bn1` rows in the same way, because the prompt gives no second source for them.

## Change from the plan

The plan used the 708 chips in `tanks-unet-256-64`. Those chips were cut around tanks, and 704 of them hold a tank. A network trained on them reached 0.98 IoU on its training pixels, but it marked about 60% of held-out pixels as tank, where the true share is 16%. That fault comes from the sampling, not the date. The source network now trains on the whole 2015 image, with the same sampling as the upper bound. Thus the date is the main difference between the two networks.

## Gates

Gate 1 uses val cells. It uses the median of 3 replies, not the best one.

| Result | Meaning | Next |
|---|---|---|
| Median does not beat no edit, or the shuffled control does not collapse | No skill | Write the result, and stop |
| Median beats no edit, and the control collapses, but below LDA and AdaBN | Skill, no practical gain | Stage C |
| Median beats LDA and AdaBN | Strong. It is not expected | Check it hard, then Stage C |

Stage C writes a tiny U-Net (10,000 to 20,000 weights) from nothing. Gate 1 skipped it. It ran later at the user's request, under its own rule.

## Commands

On gc1, from `/opt/dlami/nvme/wsp/weight-space-programming`, in tmux session `wsp`. The NVMe disk is ephemeral, so copy `results/geo/` to the Mac after each stage.

```sh
bash geo/gc1-setup.sh                                  # once
P=/opt/dlami/nvme/wsp/geo-venv/bin/python
CUDA_VISIBLE_DEVICES= $P -m unittest discover -s geo -p "test_tanks.py"
$P geo/tanks.py prepare --data /opt/dlami/nvme/wsp/tanks-data
$P geo/tanks.py train source --seed 0                  # also seeds 1 and 2, --jitter, and: train target
$P geo/tanks.py baselines
$P geo/tanks.py report
$P geo/tanks.py score <run>
$P geo/tanks.py train tiny --seed 0 --epochs 200       # Stage C
$P geo/tanks.py write-report                           # Stage C: prompt and baselines
$P geo/tanks.py write-score <run>                      # Stage C: score one reply
$P geo/tanks.py final source-s0 adabn hist-match lda-head head-retrain filter-lda source-jitter-s0 target-s0 <run>...   # once only
```

`tanks-data` holds `Tanks.tif`, `20150520_075001.tif`, `Tanks.gdb` and `split_cells.geojson`.

These use LLM capacity. Run them on the Mac, then copy the run folders to gc1:

```sh
.venv/bin/python wsp.py ask tanks repair
.venv/bin/python wsp.py ask tanks repair-shuffled
.venv/bin/python wsp.py ask tanks write
.venv/bin/python wsp.py ask tanks write --model gpt-6-astra --effort xhigh   # through the Codex login
```

## Rules

- `results/geo/` holds numbers and networks from private imagery. It stays out of git.
- A prompt holds only numbers from train cells, and no pixels. A test changes val and test pixels and labels, and the evidence must stay identical.
- `final` opens the test cells once. It refuses a second run. It ran on 2026-11-06, so these cells are no longer held out for a new design here.

## Limits

- One image pair, one area, 19 val tanks and 28 test tanks.
- Tanks that were built after 2015 are label errors in the 2015 training. The count is not measured.
- Each comparator is one network, except the no-edit row.
