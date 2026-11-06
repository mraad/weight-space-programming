

Strategy: steer toward the current target pad with a horizontal PD law, and control vertical velocity with a simple descent reference that slows near the ground and speeds up slightly as fuel drops. The first hidden layer computes two errors:

- unit 0: horizontal error `0.5 * (target_x/100 - x/100) - vx`
- unit 1: vertical error `vy + 0.2*(y/60) + 0.13 - 0.05*(fuel/300)`, which is `vy + 0.08 + 0.2*(y/60)` at full fuel

The second hidden layer saturates these errors into right/left and thrust/none action preferences. The output biases make vertical control the default, with left/right winning only when the horizontal error is large enough.

```edits
# Horizontal PD error: 0.5 * target_x_norm - 0.5 * x_norm - vx
W1[0, 0] = -0.5
W1[2, 0] = -1.0
W1[5, 0] = 0.1
W1[6, 0] = 0.25
W1[7, 0] = 0.4

# Vertical descent error: vy + 0.2*y_norm + 0.08 at full fuel
W1[1, 1] = 0.2
W1[3, 1] = 1.0
W1[4, 1] = -0.05
b1[1] = 0.13

# Saturate horizontal error for right/left selection
W2[0, 0] = 45.0
W2[1, 0] = -45.0

# Saturate vertical error for none/thrust selection
W2[2, 1] = 10.0
W2[3, 1] = -10.0

# None from positive vertical error; thrust from negative vertical error
W3[2, 0] = 6.0
W3[3, 1] = 6.0

# Left from negative horizontal error; right from positive horizontal error
W3[1, 2] = 10.0
W3[0, 3] = 10.0

# Default to vertical control; side actions need a larger error
b3[0] = 2.0
b3[1] = 2.0
b3[2] = -3.5
b3[3] = -3.5
```