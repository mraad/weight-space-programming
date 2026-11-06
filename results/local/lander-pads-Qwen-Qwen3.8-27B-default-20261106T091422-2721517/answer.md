

Design:

- Use one first-layer unit for horizontal intent:
  \(z_h = 20(x/100) + 10vx - 20\,x_\text{target}/100\).
  This is a predictive horizontal error: it is negative when the craft should go right and positive when it should go left.
- Use one first-layer unit for vertical braking:
  \(z_v = -35vy - 5\).
  This becomes positive when \(vy\) is below roughly \(-0.15\), creating a 2-no-thrust / 1-thrust descent cycle with safe touchdown speed.
- The second hidden layer only sharpens those two signed signals.
- The output layer uses:
  - default `none` logit = 0,
  - `left` = \(-9 + 27C\),
  - `right` = \(-9 - 27C\),
  - `thrust` = \(-9 + 50T\),
  so thrust has priority when needed, and horizontal actions have a small deadband.

All unlisted weights remain 0.

```edits
W1[0, 0] = 20.0
W1[2, 0] = 10.0
W1[5, 0] = -4.0
W1[6, 0] = -10.0
W1[7, 0] = -16.0
W1[3, 1] = -35.0
b1[1] = -5.0
W2[0, 0] = 1.5
W2[1, 1] = 1.5
W3[0, 2] = 27.0
W3[0, 3] = -27.0
W3[1, 1] = 50.0
b3[0] = 0.0
b3[1] = -9.0
b3[2] = -9.0
b3[3] = -9.0
```