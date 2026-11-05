## Design

**Strategy:** this is a cascaded velocity controller. Position error sets a desired horizontal speed, and altitude sets a desired descent speed. The craft holds altitude more strongly while it is far from the target, so it arrives over the pad before committing to the landing. I worked through the numbers by hand, but nothing has been run.

**First hidden layer**

| Unit | Computes | Purpose |
|---|---|---|
| 0 | `tanh(9·ex)`, where `ex = (target_x − x)/100` | Saturating position error |
| 1 | `tanh(vx)` | Approximately passes through vx |
| 2 | `tanh(vy)` | Approximately passes through vy |
| 3 | `tanh(y/60 − 0.5)` | Approximately linear altitude |
| 4 | `tanh(30·(ex − 0.08))` | Detects target more than 8 units to the right |
| 5 | `tanh(30·(−ex − 0.08))` | Detects target more than 8 units to the left |

Units 4 and 5 combine into a "far" signal: `far = 1 + (u4 + u5)/2`.

**Second hidden layer**
- **Unit 0, horizontal error:** `s = tanh(40·(0.35·u0 − u1))`. The desired horizontal speed is `0.35·tanh(9·ex)`, which cruises at about 0.35 and allows a peak braking demand of about 0.0042 per tick. That is within the roughly 0.0048 per tick available when thrust takes 40% of the ticks.
- **Unit 1, vertical error:** `t = tanh(40·(vy_des − u2))`, where `vy_des = −0.06 − 0.21·u3 + 0.09·(u4 + u5)`.
    - **Near the target:** `vy_des` runs from about −0.33 up high to −0.14 at touchdown, which leaves a good margin against the 0.25 limit.
    - **Far from the target:** it is about −0.15 up high and rises to 0 at y ≈ 12, so the craft can't touch down far from the pad.

**Outputs**
- **Logits:** none = 0, thrust = `10t`, left = `−8s − 3.5`, right = `8s − 3.5`.
- **Priority:** thrust wins once the craft is sinking about 0.012 too fast. It only needs about 40% of the ticks to hold a steady descent, so the side engines get the rest.
- **Deadband:** the −3.5 bias stops the side engines firing when vx is within about 0.012 of the desired speed, which saves fuel.

**Expected worst case:** with no target change, the longest trip (70 units) takes about 400 ticks and uses about 230 fuel. A late target change can push usage close to the 300 limit, but only in rare extreme cases.

```edits
# --- Layer 1 ---
# unit 0: saturating position error tanh(9*(target - x/100))
W1[0, 0] = -9
W1[5:8, 0] = [1.8, 4.5, 7.2]
# unit 1: vx pass-through
W1[2, 1] = 1
# unit 2: vy pass-through
W1[3, 2] = 1
# unit 3: altitude pass-through, centred
W1[1, 3] = 1
b1[3] = -0.5
# unit 4: target more than 8 units to the right
W1[0, 4] = -30
W1[5:8, 4] = [6, 15, 24]
b1[4] = -2.4
# unit 5: target more than 8 units to the left
W1[0, 5] = 30
W1[5:8, 5] = [-6, -15, -24]
b1[5] = -2.4

# --- Layer 2 ---
# unit 0: s = tanh(40*(0.35*u0 - u1)), horizontal velocity error
W2[0, 0] = 14
W2[1, 0] = -40
# unit 1: t = tanh(40*(vy_des - u2)), vertical velocity error
# vy_des = -0.06 - 0.21*u3 + 0.09*(u4 + u5)
W2[2, 1] = -40
W2[3, 1] = -8.4
W2[4, 1] = 3.6
W2[5, 1] = 3.6
b2[1] = -2.4

# --- Output layer: 0 none, 1 thrust, 2 left, 3 right ---
W3[1, 1] = 10
W3[0, 2] = -8
W3[0, 3] = 8
b3[2] = -3.5
b3[3] = -3.5
```