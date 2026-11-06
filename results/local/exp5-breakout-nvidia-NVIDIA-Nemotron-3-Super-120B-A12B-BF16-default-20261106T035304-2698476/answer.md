

```edits
# Zero all tensors (starting from zero, but explicit for clarity)
W1[:, :] = 0
b1[:] = 0
W2[:, :] = 0
b2[:] = 0
W3[:, :] = 0
b3[:] = 0

# ---- First hidden layer (unit 0 computes ball_x - paddle_x) ----
# Ball channel (ch=3): weight = column index c
W1[0*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=0
W1[1*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=1
W1[2*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=2
W1[3*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=3
W1[4*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=4
W1[5*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=5
W1[6*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=6
W1[7*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=7
W1[8*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=8
W1[9*6+3 : 600 : 60, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # c=9

# Paddle: only row 9, channel 0, weight = -column index c
W1[540 : 600 : 6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]

# Bias for unit 0 (no shift)
b1[0] = 0

# ---- Second hidden layer (identity scaling to stay in tanh linear regime) ----
# Small scaling factor on the diagonal
for i in range(32):
    W2[i, i] = 0.05
# Biases remain zero

# ---- Output layer (logits) ----
# Map unit 0 to left/right actions with gain to undo the 0.05 scaling
W3[0, 1] = -20   # left
W3[0, 3] = 20    # right
# Encourage no-op when dx ≈ 0 (ball aligned with paddle)
b3[0] = 0.5      # no-op bias
# All other logits/bias remain zero (up, down, fire, and the rest of the actions)
```