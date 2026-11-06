

Design:
- First-layer units compute thresholded features:
  - horizontal error vs. target (signed, with a small deadband)
  - velocity toward/away from target
  - vertical speed features for landing control
  - altitude features for stronger low-altitude braking
  - far-error features so the craft descends slower when still far from the target
- Second-layer units compute three action scores:
  - right score: move right when the target is to the right and not already moving right too fast
  - left score: symmetric
  - thrust score: fire main engine when falling too fast, stronger at low altitude, and slower descent when horizontally far from target
- Output layer makes the chosen score large, suppresses other actions, and prefers `none` when no score is active.

```edits
# W1: 8 x 16
# unit 0: R1 = tanh(10*(target_norm - x_norm - 0.02))
W1[0, 0] = -10
W1[5, 0] = 2
W1[6, 0] = 5
W1[7, 0] = 8
b1[0] = -0.2

# unit 1: L1 = tanh(10*(x_norm - target_norm - 0.02))
W1[0, 1] = 10
W1[5, 1] = -2
W1[6, 1] = -5
W1[7, 1] = -8
b1[1] = -0.2

# unit 2: G = tanh(5*vx)
W1[2, 2] = 5
b1[2] = 0

# unit 3: H = tanh(5*(-vx))
W1[2, 3] = -5
b1[3] = 0

# unit 4: K = tanh(8*(-vy - 0.15))
W1[3, 4] = -8
b1[4] = -1.2

# unit 5: FR = tanh(5*(target_norm - x_norm - 0.10))
W1[0, 5] = -5
W1[5, 5] = 1
W1[6, 5] = 2.5
W1[7, 5] = 4
b1[5] = -0.5

# unit 6: M = tanh(10*(vy - 0.05))
W1[3, 6] = 10
b1[6] = -0.5

# unit 7: N = tanh(10*(0.20 - y_norm))
W1[1, 7] = -10
b1[7] = 2.0

# unit 8: O = tanh(6*(0.45 - y_norm))
W1[1, 8] = -6
b1[8] = 2.7

# unit 9: FL = tanh(5*(x_norm - target_norm - 0.10))
W1[0, 9] = 5
W1[5, 9] = -1
W1[6, 9] = -2.5
W1[7, 9] = -4
b1[9] = -0.5

# W2: 16 x 16
# h2 unit 0: right score = tanh(5*(R1 - G - 0.15))
W2[0, 0] = 5
W2[2, 0] = -5
b2[0] = -0.75

# h2 unit 1: left score = tanh(5*(L1 - H - 0.15))
W2[1, 1] = 5
W2[3, 1] = -5
b2[1] = -0.75

# h2 unit 2: thrust score
# tanh(2.5*K - M + FR + FL + 0.6*N + 0.1*O + 0.7)
W2[4, 2] = 2.5
W2[6, 2] = -1
W2[5, 2] = 1
W2[9, 2] = 1
W2[7, 2] = 0.6
W2[8, 2] = 0.1
b2[2] = 0.7

# W3: 16 x 4, outputs are [none, thrust, left, right]
b3[0] = 2.5
b3[1] = -1
b3[2] = -1.5
b3[3] = -1.5

# none logit
W3[0, 0] = -5
W3[1, 0] = -5
W3[2, 0] = -6

# thrust logit
W3[2, 1] = 12

# left logit
W3[1, 2] = 15
W3[2, 2] = -10

# right logit
W3[0, 3] = 15
W3[2, 3] = -10
```