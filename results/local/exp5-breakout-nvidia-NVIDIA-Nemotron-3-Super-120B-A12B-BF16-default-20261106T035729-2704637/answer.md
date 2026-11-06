

**Design**  
The network implements a simple proportional controller:  
- Compute the horizontal error `e = paddle_x − ball_x`.  
- Pass `e` through two tanh layers (kept in the linear regime by small weights).  
- Use the final hidden unit to produce logits: left ∝ +e, right ∝ −e (no‑op and other actions get zero logit).  
Thus the paddle moves left when it is right of the ball and right when it is left of the ball, keeping the ball in play and breaking bricks.

**Weights** (all unspecified entries remain 0)

```edits
# Zero everything first
W1[:,:] = 0
b1[:] = 0
W2[:,:] = 0
b2[:] = 0
W3[:,:] = 0
b3[:] = 0

# Paddle (channel 0) weight = 0.1 * column
W1[0:600:60,0] = [0.0]*10   # c=0
W1[6:600:60,0] = [0.1]*10   # c=1
W1[12:600:60,0] = [0.2]*10  # c=2
W1[18:600:60,0] = [0.3]*10  # c=3
W1[24:600:60,0] = [0.4]*10  # c=4
W1[30:600:60,0] = [0.5]*10  # c=5
W1[36:600:60,0] = [0.6]*10  # c=6
W1[42:600:60,0] = [0.7]*10  # c=7
W1[48:600:60,0] = [0.8]*10  # c=8
W1[54:600:60,0] = [0.9]*10  # c=9

# Ball (channel 3) weight = -0.1 * column
W1[3:600:60,0] = [-0.0]*10  # c=0
W1[9:600:60,0] = [-0.1]*10  # c=1
W1[15:600:60,0] = [-0.2]*10 # c=2
W1[21:600:60,0] = [-0.3]*10 # c=3
W1[27:600:60,0] = [-0.4]*10 # c=4
W1[33:600:60,0] = [-0.5]*10 # c=5
W1[39:600:60,0] = [-0.6]*10 # c=6
W1[45:600:60,0] = [-0.7]*10 # c=7
W1[51:600:60,0] = [-0.8]*10 # c=8
W1[57:600:60,0] = [-0.9]*10 # c=9

# Hidden layer 1 -> hidden layer 2: copy unit 0
W2[0,0] = 1

# Output layer: left = +10 * h2[0], right = -10 * h2[0]
W3[0,0] = 10
W3[0,1] = -10
```
All other weights and biases stay at 0, giving the desired proportional‑controller policy.