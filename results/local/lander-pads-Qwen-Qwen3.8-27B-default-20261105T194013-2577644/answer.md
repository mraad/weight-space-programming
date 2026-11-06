

Design:
- Compute the target pad’s normalized x from the target one-hot: 0.2, 0.5, or 0.8.
- Hidden unit 0 is a vertical brake gate: it becomes positive when vy is below an altitude-dependent safe descent threshold, roughly from -0.18 high up to -0.13 near the ground.
- Hidden unit 1 is the “go right” gate: positive when the target is to the right by more than a small deadband and vx is not already large enough to the right.
- Hidden unit 2 is the symmetric “go left” gate.
- The second hidden layer amplifies these three gates.
- The output layer gives priority to thrust when the vertical gate is strong, otherwise to the active side gate, otherwise to none. Large logit margins make sampled actions and sticky actions robust.

```edits
# Vertical gate: h1[0] = tanh(20 * (-0.13 - 0.05*y_norm - vy))
W1[1, 0] = -1.0
W1[3, 0] = -20.0
b1[0] = -2.6

# Right gate: h1[1] = tanh(10 * (target_x - x - 0.02 - 0.7*vx))
W1[0, 1] = -10.0
W1[2, 1] = -7.0
W1[5:8, 1] = [2.0, 5.0, 8.0]
b1[1] = -0.2

# Left gate: h1[2] = tanh(10 * (x - target_x - 0.02 + 0.7*vx))
W1[0, 2] = 10.0
W1[2, 2] = 7.0
W1[5:8, 2] = [-2.0, -5.0, -8.0]
b1[2] = -0.2

# Amplify the three gates in the second hidden layer
W2[0, 0] = 3.0
W2[1, 1] = 6.0
W2[2, 2] = 6.0

# Output logits:
# thrust = 30*h2[0] - 10
# left   = 15*h2[2] - 5
# right  = 15*h2[1] - 5
# none   = 0
W3[0, 1] = 30.0
b3[1] = -10.0
W3[2, 2] = 15.0
b3[2] = -5.0
W3[1, 3] = 15.0
b3[3] = -5.0
b3[0] = 0.0
```