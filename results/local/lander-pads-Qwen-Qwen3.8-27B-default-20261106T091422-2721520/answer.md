

The policy is a hand-built proportional/velocity controller:

- Hidden unit 0: horizontal error to the target, roughly \((x - \text{target}_x)/20\), using the target one-hot.
- Hidden unit 1: scaled horizontal velocity \(8vx\).
- Hidden unit 2: vertical urgency, positive when \(vy\) is below about \(-0.17\).
- Hidden layer 2 passes these three units forward.
- Output logits:
  - left/right: PD control toward the target, with a deadband so “none” wins when close and slow.
  - thrust: fires when vertical velocity is too negative.
  - none: neutral baseline.

```edits
W1[0, 0] = 5
W1[5, 0] = -1
W1[6, 0] = -2.5
W1[7, 0] = -4
W1[2, 1] = 8
W1[3, 2] = -10
b1[2] = -1.7
W2[0:3, 0:3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
W3[0, 2] = 8
W3[0, 3] = -8
W3[1, 2] = 6
W3[1, 3] = -6
W3[2, 1] = 16
b3[0] = 0
b3[1] = -1
b3[2] = -2.5
b3[3] = -2.5
```