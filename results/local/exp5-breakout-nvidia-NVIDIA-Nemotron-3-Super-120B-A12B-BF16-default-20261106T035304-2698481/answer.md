

```edits
# First hidden layer: compute ball_x - paddle_x (alpha = 1)
# Paddle channel (channel 0) -> weight = -col
# Ball channel (channel 3)   -> weight = +col
# All other channels weight 0
# Only hidden unit 0 is used; others stay zero.

# Row 0
W1[0:60:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[3:60:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 1
W1[60:120:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[63:120:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 2
W1[120:180:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[123:180:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 3
W1[180:240:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[183:240:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 4
W1[240:300:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[243:300:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 5
W1[300:360:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[303:360:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 6
W1[360:420:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[363:420:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 7
W1[420:480:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[423:480:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 8
W1[480:540:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[483:540:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Row 9
W1[540:600:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[543:600:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Biases for first hidden layer (all zero)
b1[0:32] = 0

# Second hidden layer: small linear pass-through (epsilon = 0.1) on unit 0
W2[0, 0] = 0.1
# All other W2 entries remain zero; biases zero
b2[0:32] = 0

# Output layer: map hidden unit 0 to left/right logits, bias for no‑op
gain = 20.0
W3[0, 1] = -gain   # left
W3[0, 3] =  gain   # right
# All other W3 entries stay zero

# Biases: encourage no‑op when ball and paddle aligned
b3[0] = 1.0   # no‑op bias
# b3[1..5] remain zero
```