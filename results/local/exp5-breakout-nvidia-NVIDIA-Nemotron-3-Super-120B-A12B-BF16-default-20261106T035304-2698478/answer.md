

**Design description**

The network uses a single hidden unit to compute the signed horizontal distance between the ball and the paddle.  
- Hidden unit 0 receives a small positive weight ( 0.05 ) from the ball’s x‑coordinate (channel 3) and the same magnitude negative weight from the paddle’s x‑coordinate (channel 0). Its pre‑activation is therefore 0.05·(ball_x − paddle_x).  
- Because the input values are at most 9, the pre‑activation stays in [−0.45, 0.45]; tanh is almost linear here, so the unit’s output ≈ 0.05·(ball_x − paddle_x).  
- A second hidden layer simply copies this value (identity weights on the first unit).  
- The output layer maps this value to the action logits: a large negative weight (−20) to the left‑action logit and a large positive weight (+20) to the right‑action logit. This yields logits ≈ paddle_x − ball_x for left and ≈ ball_x − paddle_x for right.  
- Biases suppress the up, down, and fire actions (large negative value) and leave the no‑op logit at zero. When the ball is aligned with the paddle the three relevant logits are zero, giving an equal chance to move left, right, or stay – a harmless tie‑breaking behaviour.  

All other weights are left at their initial zero value, so they do not interfere.

```edits
# Hidden unit 0: ball_x - paddle_x (scaled by 0.05)
W1[0*6+3 : 600 : 60, 0] = 0.05 * 0
W1[1*6+3 : 600 : 60, 0] = 0.05 * 1
W1[2*6+3 : 600 : 60, 0] = 0.05 * 2
W1[3*6+3 : 600 : 60, 0] = 0.05 * 3
W1[4*6+3 : 600 : 60, 0] = 0.05 * 4
W1[5*6+3 : 600 : 60, 0] = 0.05 * 5
W1[6*6+3 : 600 : 60, 0] = 0.05 * 6
W1[7*6+3 : 600 : 60, 0] = 0.05 * 7
W1[8*6+3 : 600 : 60, 0] = 0.05 * 8
W1[9*6+3 : 600 : 60, 0] = 0.05 * 9
W1[0*6+0 : 600 : 60, 0] = -0.05 * 0
W1[1*6+0 : 600 : 60, 0] = -0.05 * 1
W1[2*6+0 : 600 : 60, 0] = -0.05 * 2
W1[3*6+0 : 600 : 60, 0] = -0.05 * 3
W1[4*6+0 : 600 : 60, 0] = -0.05 * 4
W1[5*6+0 : 600 : 60, 0] = -0.05 * 5
W1[6*6+0 : 600 : 60, 0] = -0.05 * 6
W1[7*6+0 : 600 : 60, 0] = -0.05 * 7
W1[8*6+0 : 600 : 60, 0] = -0.05 * 8
W1[9*6+0 : 600 : 60, 0] = -0.05 * 9

# Copy the first hidden unit through the second layer (identity on unit 0)
W2[0,0] = 1.0
W2[1,1] = 1.0   # unit 1 unused but kept harmless

# Output layer: map unit 0 to left/right actions with strong opposite weights
W3[0,1] = -20.0   # left action logit
W3[0,3] = 20.0    # right action logit

# Suppress useless actions (up, down, fire) with large negative bias
b3[2] = -10.0   # up
b3[4] = -10.0   # down
b3[5] = -10.0   # fire
# (no‑op, left, right biases remain at their default 0)
```