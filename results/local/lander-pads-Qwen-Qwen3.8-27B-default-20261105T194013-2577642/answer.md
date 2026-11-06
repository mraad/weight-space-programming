

Design:
- First hidden layer creates compact features: a saturated target-minus-x error, vx, vy, a low-altitude feature, and a higher-gain vx feature for braking.
- Second hidden layer forms:
  - a right command: positive when target is to the right and vx is not already too positive;
  - a left command: the opposite;
  - a vertical command: thrust when vy is below an altitude-dependent safe descent threshold;
  - a low-altitude gate that softens vertical priority near the ground so horizontal correction can still happen.
- Output logits default to none, boost thrust on vertical error, and boost left/right on the horizontal PD signal.

```edits
# h1: saturated target-x error, vx, vy, low altitude, high vx
W1[0, 0] = -2.0
W1[5, 0] = 0.4
W1[6, 0] = 1.0
W1[7, 0] = 1.6
W1[2, 1] = 1.0
W1[3, 2] = 1.0
W1[1, 3] = -10.0
W1[2, 4] = 3.0
b1[3] = 1.0

# h2: right, left, vertical thrust, low-altitude gate
W2[0, 0] = 4.5
W2[1, 0] = -5.0
W2[4, 0] = -0.8
W2[0, 1] = -4.5
W2[1, 1] = 5.0
W2[4, 1] = 0.8
W2[3, 2] = 0.204
W2[2, 2] = -6.0
b2[2] = -0.396
W2[3, 3] = 1.0
b2[3] = 1.0

# outputs: none, thrust, left, right
b3[0:4] = [1.5, -1.0, -1.0, -1.0]
W3[0, 3] = 8.0
W3[1, 2] = 8.0
W3[2, 1] = 18.0
W3[3, 1] = -4.0
```