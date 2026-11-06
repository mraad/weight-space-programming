

Design: use a hand-tuned probabilistic controller.  
- Hidden layer 1 computes: right/left horizontal error terms `target_x - x` and `x - target_x`, with velocity feedforward `±2*vx` and a small dead zone; a vertical velocity feature centered at `vy = -0.1`; and an extra low-vy safety feature.  
- Hidden layer 2 smooths these into action commands.  
- Output logits: side actions are low unless the corresponding horizontal command is positive; `none` and `thrust` are biased so that at `vy ≈ -0.1`, `thrust` is sampled about 38–40% of the time, giving a slow, safe descent. As `vy` becomes more negative, `thrust` probability rises sharply.

```edits
# h1 column 0: right horizontal command pre = 10 * ((target_x - x) - 2*vx - 0.02)
W1[0:8, 0] = [-10.0, 0.0, -20.0, 0.0, 0.0, 2.0, 5.0, 8.0]
b1[0] = -0.2

# h1 column 1: left horizontal command pre = 10 * ((x - target_x) + 2*vx - 0.02)
W1[0:8, 1] = [10.0, 0.0, 20.0, 0.0, 0.0, -2.0, -5.0, -8.0]
b1[1] = -0.2

# h1 column 2: vertical velocity feature, zero at vy = -0.1
W1[0:8, 2] = [0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0]
b1[2] = 0.8

# h1 column 14: very-low vertical speed feature, centered near vy = -0.18
W1[0:8, 14] = [0.0, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0, 0.0]
b1[14] = -1.44

# h2 column 0: smoothed vertical control
W2[2, 0] = 2.0

# h2 column 1: right command
W2[0, 1] = 4.0

# h2 column 2: left command
W2[1, 2] = 4.0

# h2 column 3: extra thrust when vy is below the normal landing speed
W2[14, 3] = 3.0
b2[3] = 1.7

# logits: side actions are low unless commanded; none/thrust differ by 0.5 at vy=-0.1
W3[0, 0] = 3.0
W3[0, 1] = -3.0
W3[1, 3] = 6.0
W3[2, 2] = 6.0
W3[3, 1] = 1.0

b3[0] = 2.0
b3[1] = 1.5
b3[2] = -3.0
b3[3] = -3.0
```