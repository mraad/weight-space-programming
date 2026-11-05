## Strategy

I use a cascaded controller, with a target speed for each axis and engines firing to track it. Fuel is enough, so I don't use the fuel input. I worked out the numbers by hand and haven't run the network, so its score is untested.

**Horizontal control:** the desired speed is `vx_des = 0.4·tanh(E/20)`, where E is the distance to the target in game units.
- The craft cruises at about 0.4 when far away and slows smoothly on approach.
- The braking this requires peaks at about 0.003 per tick. That's within the roughly 0.0046 per tick available when side thrust shares ticks with the main engine.
- Left and right fire only when `vx` misses `vx_des` by more than about 0.025, so little fuel is wasted. Near the pad, `|vx|` stays below 0.1.

**Vertical control:** thrust fires when `vy` is below a desired descent speed, which depends on altitude and whether the craft is within about 3 units of the target:

| | Within 3 units of target | Farther away |
|---|---|---|
| **High (y > ~10)** | -0.30 | -0.15 |
| **Low (y < ~10)** | -0.12 (gentle landing) | +0.03 (holds at y ≈ 8) |

The craft therefore can't touch down off-pad, and it descends only once it is lined up. The desired speed is a linear sum of the "far" and "low" signals, so no separate AND unit is needed. The high thrust gain keeps `vy` within about ±0.03 of the desired speed. Landing speed is about 0.12, well inside the 0.25 limit.

**Worst-case estimate:** for a 70-unit traverse starting with the wrong velocity, the craft lands around tick 400. It uses about 230 of its 300 fuel. A late target switch across the map can use close to all 300, so a few of those episodes may run out.

## Hidden units

**Layer 1 (unused units stay at 0):**
- `h1[0] = tanh(5·e)`, where `e = x_target − x` in input units: the saturated position error
- `h1[1] ≈ 0.5·vx`
- `h1[2] ≈ 0.5·vy`
- `h1[3] = tanh(60·(e − 0.03))`: target is to the right, beyond 3 units
- `h1[4] = tanh(60·(−e − 0.03))`: target is to the left, beyond 3 units
- `h1[5] = tanh(0.5·(10 − y))`: low altitude

**Layer 2:**
- `h2[0] = tanh(60·(vx_des − vx − 0.025))`: fire right
- `h2[1] = tanh(60·(vx − vx_des − 0.025))`: fire left
- `h2[2] = tanh(20·(vy_des − vy))`: fire thrust, with `vy_des = −0.06 + 0.075·(h1[3] + h1[4]) + 0.09·h1[5]`

**Outputs:**
- The "none" logit stays at 0.
- The thrust logit is `6·h2[2]`.
- The left logit is `4·h2[1] − 1`.
- The right logit is `4·h2[0] − 1`.

Thrust and side logits compete through the softmax. When both are demanded, thrust settles at the 40% duty needed to hold `vy`, leaving about 57% for the side engines. If `vy` falls well below the desired speed, thrust takes priority.

```edits
# Layer 1
# unit 0: saturated position error tanh(5*(x_target - x))
W1[0, 0] = -5
W1[5:8, 0] = [1, 2.5, 4]
# unit 1: 0.5*vx (near-linear)
W1[2, 1] = 0.5
# unit 2: 0.5*vy (near-linear)
W1[3, 2] = 0.5
# unit 3: target to the right by more than 3 units
W1[0, 3] = -60
W1[5:8, 3] = [12, 30, 48]
b1[3] = -1.8
# unit 4: target to the left by more than 3 units
W1[0, 4] = 60
W1[5:8, 4] = [-12, -30, -48]
b1[4] = -1.8
# unit 5: low altitude (y < 10)
W1[1, 5] = -30
b1[5] = 5

# Layer 2
# unit 0: fire right
W2[0, 0] = 24
W2[1, 0] = -120
b2[0] = -1.5
# unit 1: fire left
W2[0, 1] = -24
W2[1, 1] = 120
b2[1] = -1.5
# unit 2: fire main engine
W2[2, 2] = -40
W2[3, 2] = 1.5
W2[4, 2] = 1.5
W2[5, 2] = 1.8
b2[2] = -1.2

# Outputs: 0 none, 1 thrust, 2 left, 3 right
W3[2, 1] = 6
W3[1, 2] = 4
W3[0, 3] = 4
b3[2] = -1
b3[3] = -1
```