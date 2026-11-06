

**Design description**

The network uses the first hidden layer to compute four useful, small‑scale features:

* Unit 0 → 0.5 × ( target_x − x ) / 100  (horizontal error, positive when craft is left of the target)
* Unit 1 → −0.5 × vx  (oppose horizontal velocity)
* Unit 2 → −0.5 × vy  (oppose vertical velocity; positive when falling down)
* Unit 3 → 0.5 × fuel / 300  (remaining fuel)

All other hidden units are kept at zero.  
The second hidden layer is an identity mapping (weights = 1 on the diagonal, bias = 0) so that the features are passed through with only the mild tanh non‑linearity (the features are kept in the ±0.5 range, where tanh is almost linear).  

The output layer turns these features into action logits:

* **Thrust** (action 1) gets a positive weight from unit 2 (‑0.5 × vy) – we thrust when the craft is falling fast.
* **Left** (action 2) gets a negative weight from unit 0 (error) – we fire left when the error is negative (craft right of target).
* **Right** (action 3) gets a positive weight from unit 0 – we fire right when the error is positive (craft left of target).
* **None** (action 0) has a small baseline bias; the other actions have a negative bias so that “none” is chosen when no correction is needed.

Biases are set to give a modest preference for doing nothing when the craft is already near the target and moving slowly.

All unspecified weights remain zero, as required.

```edits
W1[0,0] = -0.5
W1[5,0] = 0.1
W1[6,0] = 0.25
W1[7,0] = 0.4
W1[2,1] = -0.5
W1[3,2] = -0.5
W1[4,3] = 0.5
# b1 remains zero (default)

W2[0,0] = 1
W2[1,1] = 1
W2[2,2] = 1
W2[3,3] = 1
W2[4,4] = 1
W2[5,5] = 1
W2[6,6] = 1
W2[7,7] = 1
W2[8,8] = 1
W2[9,9] = 1
W2[10,10] = 1
W2[11,11] = 1
W2[12,12] = 1
W2[13,13] = 1
W2[14,14] = 1
W2[15,15] = 1
# other W2 entries stay zero
# b2 stays zero (default)

W3[0,2] = -4.0
W3[0,3] = 4.0
W3[2,1] = 4.0
# other W3 entries stay zero

b3[0] = 0.0
b3[1] = -0.5
b3[2] = -0.5
b3[3] = -0.5
```