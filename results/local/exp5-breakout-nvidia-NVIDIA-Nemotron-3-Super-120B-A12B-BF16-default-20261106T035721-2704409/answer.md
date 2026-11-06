

```edits
# Policy: move paddle left/right to follow the ball's x‑coordinate.
# s = 0.05 scales coordinates into the linear range of tanh.
# Hidden unit 0 → s * ball_x, hidden unit 1 → s * paddle_x.
# W2 passes these two units unchanged.
# Output layer: right logit = (1/s)*(h1[0] - h1[1]), left logit = -(1/s)*(h1[0] - h1[1]).
# With s=0.05, 1/s = 20.

# Ball channel (channel 3) → hidden unit 0
W1[3:600:60, 0]   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 0
W1[9:600:60, 0]   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 1
W1[15:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 2
W1[21:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 3
W1[27:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 4
W1[33:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 5
W1[39:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 6
W1[45:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 7
W1[51:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 8
W1[57:600:60, 0]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # column 9

# Paddle channel (channel 0, row 9) → hidden unit 1
W1[540:600:6, 1]  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # columns 0‑9

# Hidden layer 2: identity for the first two units (others stay zero)
W2[0, 0] = 1
W2[1, 1] = 1

# Output layer: map hidden units 0 & 1 to left (action 1) and right (action 3) logits
W3[0, 1] = -20   # h1[0] → left logit
W3[1, 1] =  20   # h1[1] → left logit
W3[0, 3] =  20   # h1[0] → right logit
W3[1, 3] = -20   # h1[1] → right logit

# All biases remain zero (default).
```