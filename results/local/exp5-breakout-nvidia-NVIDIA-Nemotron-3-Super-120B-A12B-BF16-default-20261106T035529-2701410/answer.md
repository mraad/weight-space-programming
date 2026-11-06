

**Design description**  
The network uses a single hidden unit (unit 0 of the first layer) to compute the signed horizontal distance between the ball and the paddle:  

- For every cell that contains the ball (channel 3) we add its column index.  
- For every cell that contains the paddle (channel 0, only on row 9) we subtract its column index.  

All other inputs are weighted zero, and the bias of this unit is 0. After a tanh non‑linearity the unit’s activation is ≈ tanh(ball_x − paddle_x), i.e. a smooth sign of the difference.  

The second hidden layer simply copies this value (weight = 1).  

The output layer turns the copied value into logits for the left and right actions:  

- left logit = −2 · h₂[0]  
- right logit = +2 · h₂[0]  

A bias of +1 is given to the no‑op action so that when the ball is roughly above the paddle (difference near 0) the network prefers to do nothing. All other actions (up, down, fire) receive zero logits. With the sticky‑action probability (0.1) this simple heuristic still learns to move the paddle toward the falling ball and break most bricks.

---

```edits
# Ball channel weights for hidden unit 0 (add column index)
W1[3:600:60,0] = 0
W1[9:600:60,0] = 1
W1[15:600:60,0] = 2
W1[21:600:60,0] = 3
W1[27:600:60,0] = 4
W1[33:600:60,0] = 5
W1[39:600:60,0] = 6
W1[45:600:60,0] = 7
W1[51:600:60,0] = 8
W1[57:600:60,0] = 9

# Paddle channel weights for hidden unit 0 (subtract column index)
W1[540,0] = 0
W1[546,0] = -1
W1[552,0] = -2
W1[558,0] = -3
W1[564,0] = -4
W1[570,0] = -5
W1[576,0] = -6
W1[582,0] = -7
W1[588,0] = -8
W1[594,0] = -9

# Copy first hidden unit to second hidden unit 0
W2[0,0] = 1

# Output layer: map h2[0] to left/right logits
W3[0,1] = -2   # left
W3[0,3] =  2   # right

# Bias to favour no-op when the ball is centred
b3[0] = 1
```