I worked out these weights by hand and traced a few states through the network, but nothing has been simulated, so the performance is unmeasured.

## Strategy
The network runs two independent speed controllers:

- **Horizontal:** it steers toward a desired `vx` that depends on the distance `E` to the target. This speed is about 0.43 when far away, 0.30 at `E = 20`, 0.10 at `E = 3` and 0 at the pad. The braking this needs is at most about 0.0026 per tick, well under the roughly 0.0048 available, so it shouldn't overshoot.
- **Vertical:** it thrusts when `vy` is below a desired value:
  - **Aligned:** the desired value is `vy_des = -(0.13 + 0.4·y/60)`, which gives a touchdown speed of about 0.13–0.17 (the limit is 0.25).
  - **Off target or moving fast:** if `|E| > 3` or `|vx| > 0.07`, the desired value rises by up to 0.26. The craft then descends slowly toward a holding height of about 20, so it can't touch down until it's lined up.
- **Action priority:** thrust has the largest logit (±12), side thrust is next (±8) and doing nothing is 0. High gains keep the choice close to deterministic.

**Fuel estimate:** the worst case for the main score (70 units of travel with `vx` initially pointing away) takes about 430 ticks and uses about 226 of the 300 fuel. A target switch at tick 120 is estimated at about 340 ticks to reach the new pad, still with fuel to spare.

## Hidden units

| Unit | Computes | Purpose |
|---|---|---|
| Layer 1, unit 0 | `tanh(3.5·e)`, where `e = (target_x − x)/100` | Broad part of the desired `vx` |
| Layer 1, unit 1 | `tanh(25·e)` | Fine part of the desired `vx` near the pad |
| Layer 1, unit 2 | `tanh(0.5·vx)` | Roughly linear copy of `vx` |
| Layer 1, unit 3 | `tanh(0.5·(vy + 0.4·y/60 + 0.13))` | Vertical speed error when aligned |
| Layer 1, units 4 and 5 | `e > 0.03`, `e < −0.03` | Off-target detectors |
| Layer 1, units 6 and 7 | `vx > 0.07`, `vx < −0.07` | Too-fast detectors |
| Layer 2, unit 0 | `tanh(60·(s − 0.015))`, where `s` = desired `vx` − `vx` | Fire right, with a small dead band |
| Layer 2, unit 1 | `tanh(60·(−s − 0.015))` | Fire left |
| Layer 2, unit 2 | `tanh(40·(vy_des − vy))` | Fire main thrust |

The dead band limits wasted fuel from small side corrections, and it lets `x` settle within about 0.5 units of the pad center.

```edits
# Layer 1: position error at two scales
W1[0:8, 0] = [-3.5, 0, 0, 0, 0, 0.7, 1.75, 2.8]
W1[0:8, 1] = [-25, 0, 0, 0, 0, 5, 12.5, 20]

# Layer 1: vx copy
W1[2, 2] = 0.5

# Layer 1: vertical speed error
W1[1, 3] = 0.2
W1[3, 3] = 0.5
b1[3] = 0.065

# Layer 1: off-target detectors (|E| > 3)
W1[0:8, 4] = [-100, 0, 0, 0, 0, 20, 50, 80]
b1[4] = -3
W1[0:8, 5] = [100, 0, 0, 0, 0, -20, -50, -80]
b1[5] = -3

# Layer 1: too-fast detectors (|vx| > 0.07)
W1[2, 6] = 60
b1[6] = -4.2
W1[2, 7] = -60
b1[7] = -4.2

# Layer 2: fire right
W2[0:3, 0] = [19.8, 6, -120]
b2[0] = -0.9

# Layer 2: fire left
W2[0:3, 1] = [-19.8, -6, 120]
b2[1] = -0.9

# Layer 2: main thrust, with the descent limit from the detectors
W2[3:8, 2] = [-80, 3.2, 3.2, 2, 2]
b2[2] = 10.4

# Outputs: 0 none stays at 0, 1 thrust, 2 left, 3 right
W3[0, 3] = 8
W3[1, 2] = 8
W3[2, 1] = 12
```