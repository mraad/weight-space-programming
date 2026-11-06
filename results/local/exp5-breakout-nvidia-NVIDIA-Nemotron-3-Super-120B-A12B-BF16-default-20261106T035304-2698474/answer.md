

**Design description**

The network first extracts the horizontal position of the paddle (channel 0, row 9) and the ball (channel 3) by weighting each cell with its column index.  
Two hidden units (unit 0 and unit 1) compute small‑amplitude versions of `paddle_x` and `ball_x` (weights = 0.01·column, kept in the linear region of tanh).  
A second hidden unit (unit 0 of the second layer) forms the difference `ball_x − paddle_x` via weights −0.1 and +0.1 on the first‑layer units (again staying in tanh’s linear region).  
The output layer turns this difference into logits: a large negative weight (‑1000) sends strong left‑logit when the ball is left of the paddle, a large positive weight (+1000) sends strong right‑logit when the ball is right, and a bias of +1 on the no‑op logit makes the paddle stay still when aligned. All other actions receive zero logits.  
Because the weights are small enough that tanh≈identity, the network behaves essentially as the desired linear controller.

```
edits
# Paddle feature (unit 0): weight = 0.01 * column index
W1[540:600:6, 0] = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# Ball feature (unit 1): weight = 0.01 * column index for every row
W1[3:600:60, 1] = [0.00]*10   # column 0
W1[9:600:60, 1] = [0.01]*10   # column 1
W1[15:600:60, 1] = [0.02]*10  # column 2
W1[21:600:60, 1] = [0.03]*10  # column 3
W1[27:600:60, 1] = [0.04]*10  # column 4
W1[33:600:60, 1] = [0.05]*10  # column 5
W1[39:600:60, 1] = [0.06]*10  # column 6
W1[45:600:60, 1] = [0.07]*10  # column 7
W1[51:600:60, 1] = [0.08]*10  # column 8
W1[57:600:60, 1] = [0.09]*10  # column 9

# Second hidden layer: compute ball_x - paddle_x (scaled)
W2[0, 0] = -0.1
W2[1, 0] = 0.1
b2[:] = 0                     # zero bias for second layer

# Output layer: map difference to logits
W3[0, 0] = -1000              # left logit  = -diff
W3[0, 2] = 1000               # right logit = +diff
b3[:] = 0
b3[5] = 1                     # bias for no‑op (stay still when aligned)
```