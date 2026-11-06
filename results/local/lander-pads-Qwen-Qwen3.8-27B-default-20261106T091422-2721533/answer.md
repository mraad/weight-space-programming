

Design:
- Vertical controller: thrust when `vy` is below about -0.15; otherwise let gravity pull down. This keeps touchdown speed safe and uses fuel efficiently.
- Horizontal controller: compute normalized error to the currently selected pad, subtract a predictive term `0.65 * vx`, and use a small deadband. Right if this is positive, left if negative.
- Speed clamp: force the opposite side thrust when `|vx|` exceeds about 0.6, so the craft does not approach the target with excessive horizontal speed.
- Priority: vertical thrust has priority when `vy` is too fast; otherwise the speed clamp and horizontal controller choose side/none.

Hidden units used:
- `h1[:, 0]`: right horizontal control preactivation.
- `h1[:, 1]`: left horizontal control preactivation.
- `h1[:, 2]`: vertical “thrust if `vy` too fast” preactivation.
- `h1[:, 3]`: high-positive-velocity clamp (`vx > 0.6`), feeds left.
- `h1[:, 4]`: high-negative-velocity clamp (`vx < -0.6`), feeds right.
- `h2` sharpens these signals; `W3` maps them to action logits.

```edits
W1[0:8, 0] = [-50.0, 0.0, -32.5, 0.0, 0.0, 10.0, 25.0, 40.0]
b1[0] = -0.25
W1[0:8, 1] = [50.0, 0.0, 32.5, 0.0, 0.0, -10.0, -25.0, -40.0]
b1[1] = -0.25
W1[0:8, 2] = [0.0, 0.0, 0.0, -50.0, 0.0, 0.0, 0.0, 0.0]
b1[2] = -7.5
W1[0:8, 3] = [0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0]
b1[3] = -30.0
W1[0:8, 4] = [0.0, 0.0, -50.0, 0.0, 0.0, 0.0, 0.0, 0.0]
b1[4] = -30.0
W2[0, 0] = 10.0
W2[1, 1] = 10.0
W2[2, 2] = 10.0
W2[3, 3] = 10.0
W2[4, 4] = 10.0
W3[0, 3] = 40.0
W3[1, 2] = 40.0
W3[2, 1] = 120.0
W3[3, 2] = 80.0
W3[4, 3] = 80.0
b3[0:4] = [0.0, -20.0, -20.0, -20.0]
```