

Design summary:

- First hidden units:
  - Unit 0: `tanh(3 * (target_x/100 - x/100))` — signed horizontal error to the target.
  - Unit 1: `tanh(3 * vx)` — current horizontal velocity, scaled to match the error units.
  - Unit 2: `tanh(4 * vy)` — current vertical velocity.
- Second hidden units:
  - Unit 0: `tanh(6 * (unit0 - unit1))` — horizontal demand: positive means push right, negative means push left.
  - Unit 1: `tanh(-15 * unit2 - 7.2)` — vertical braking demand: becomes positive when `vy` is too negative.
- Outputs:
  - `none` is the baseline.
  - `thrust` is driven strongly by the vertical braking unit.
  - `left` and `right` are driven oppositely by the horizontal demand unit.

This makes the craft descend slowly, pulse main thrust to cap vertical speed, and use side thrust to track the target with proportional braking. The target one-hot is used directly in the first-layer error computation, so it reacts immediately if the target changes.

```edits
W1[0,0] = -3.0
W1[5,0] = 0.6
W1[6,0] = 1.5
W1[7,0] = 2.4
W1[2,1] = 3.0
W1[3,2] = 4.0

W2[0,0] = 6.0
W2[1,0] = -6.0
W2[2,1] = -15.0
b2[0] = 0.0
b2[1] = -7.2

W3[0,2] = -15.0
W3[0,3] = 15.0
W3[1,1] = 30.0
b3[0] = 0.0
b3[1] = -10.0
b3[2] = -6.0
b3[3] = -6.0
```