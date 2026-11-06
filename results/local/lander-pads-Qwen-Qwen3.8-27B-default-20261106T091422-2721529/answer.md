

Design (short):
- Use the first hidden layer to compute a small set of control signals:
  - unit 0: horizontal “move right” signal: tanh(10 * (target_x/100 - x/100) - 12 * vx)
  - unit 1: exact negative of unit 0, so it becomes “move left”
  - unit 2: vertical soft-landing signal: tanh(20 * (-vy - 0.125)), i.e. fire main thrust when downward velocity is too large
- Use the second hidden layer only to sharpen these three signals with tanh(4 * h1).
- Output logits:
  - none is the default with bias 0
  - left/right use large negative bias + positive weight from the horizontal signals
  - thrust uses a larger weight from the vertical signal so vertical safety wins when needed
- This implements a PD-style policy: steer toward the target with velocity damping, and throttle to keep vy around roughly -0.12 to -0.14.

```edits
# Horizontal right feature: pre = 10*(target_norm - x_norm) - 12*vx
W1[0, 0] = -10.0
W1[2, 0] = -12.0
W1[5, 0] = 2.0
W1[6, 0] = 5.0
W1[7, 0] = 8.0
b1[0] = 0.0

# Horizontal left feature: exact negative of unit 0
W1[0, 1] = 10.0
W1[2, 1] = 12.0
W1[5, 1] = -2.0
W1[6, 1] = -5.0
W1[7, 1] = -8.0
b1[1] = 0.0

# Vertical thrust feature: pre = 20*(-vy - 0.125)
W1[3, 2] = -20.0
b1[2] = -2.5

# Second layer: sharpen the three features
W2[0, 0] = 4.0
W2[1, 1] = 4.0
W2[2, 2] = 4.0

# Outputs: h2[0] -> right, h2[1] -> left, h2[2] -> thrust
W3[0, 3] = 50.0
W3[1, 2] = 50.0
W3[2, 1] = 80.0

# Biases: none is default; actions must clearly activate
b3[0] = 0.0
b3[1] = -45.0
b3[2] = -45.0
b3[3] = -45.0
```