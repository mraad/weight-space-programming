

**Design description**

The network implements a simple “chase‑the‑ball” policy:

1. **First hidden layer (units 0‑1)**  
   *Unit 0* computes a (scaled) sum of the ball’s ×‑coordinate: for every cell that contains the ball (channel 3) we add `0.1 × column`.  
   *Unit 1* computes the same for the paddle (channel 0, row 9).  
   Because the activation function is tanh and the pre‑activations are kept small (|·|≤0.9), tanh≈identity, so the units encode `0.1·ball_x` and `0.1·paddle_x`.

2. **Second hidden layer (unit 0)**  
   Takes the difference `paddle_x − ball_x` by weighting unit 1 with +10 and unit 0 with −10 (the factors cancel the 0.1 scaling).  
   All other hidden units are unused and kept at zero.

3. **Output layer**  
   The difference `d = h2[0]` (approximately `paddle_x − ball_x`) is turned into logits:  
   * left → `+k·d`  
   * right → `−k·d`  
   * no‑op → small bias (0.5) to keep the paddle still when centered.  
   With `k=2` the logits are large enough to dominate the softmax, yielding left when the ball is left of the paddle, right when it is right, and no‑op when aligned.

All other actions (up, down, fire) receive zero logits.

---

```edits
# First hidden layer: ball_x (unit 0) and paddle_x (unit 1)
# Ball: for each column c, add 0.1*c to all rows (channel 3)
W1[3:600:60, 0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # placeholder, will be overwritten per column
W1[9:600:60, 0] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
W1[15:600:60, 0] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
W1[21:600:60, 0] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
W1[27:600:60, 0] = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
W1[33:600:60, 0] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
W1[39:600:60, 0] = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
W1[45:600:60, 0] = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
W1[51:600:60, 0] = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
W1[57:600:60, 0] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

# Paddle: only row 9, channel 0
W1[540, 1] = 0.0   # column 0
W1[546, 1] = 0.1   # column 1
W1[552, 1] = 0.2   # column 2
W1[558, 1] = 0.3   # column 3
W1[564, 1] = 0.4   # column 4
W1[570, 1] = 0.5   # column 5
W1[576, 1] = 0.6   # column 6
W1[582, 1] = 0.7   # column 7
W1[588, 1] = 0.8   # column 8
W1[594, 1] = 0.9   # column 9

# Biases for first layer (keep zero)
b1[0] = 0
b1[1] = 0

# Second hidden layer: compute paddle_x - ball_x (unit 0)
W2[0, 0] = -10.0   # from ball unit
W2[1, 0] =  10.0   # from paddle unit
b2[0] = 0.0

# Output layer: map difference to logits
W3[0, 1] =  2.0    # left  (action 1)
W3[0, 3] = -2.0    # right (action 3)
b3[0] = 0.5        # no-op bias (action 0)
# All other weights and biases remain zero (default)
```