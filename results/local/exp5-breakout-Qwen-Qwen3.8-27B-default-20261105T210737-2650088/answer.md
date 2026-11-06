

Design: use a fixed, robust tracking policy: keep the paddle in the ball’s column.  
- First-layer unit 0 computes \(4(\text{ball}_x-\text{paddle}_x)-2\), so it becomes positive when the ball is to the right of the paddle.  
- First-layer unit 1 computes \(-4(\text{ball}_x-\text{paddle}_x)-2\), so it becomes positive when the ball is to the left of the paddle.  
- The second layer sharply amplifies these two indicators.  
- The output layer decodes them: if the ball is right, output right; if left, output left; if aligned, output no-op. Other actions are strongly suppressed.

```edits
# unit 0: right indicator, tanh(4*(ball_x - paddle_x) - 2)
W1[9:600:60, 0] = 4
W1[15:600:60, 0] = 8
W1[21:600:60, 0] = 12
W1[27:600:60, 0] = 16
W1[33:600:60, 0] = 20
W1[39:600:60, 0] = 24
W1[45:600:60, 0] = 28
W1[51:600:60, 0] = 32
W1[57:600:60, 0] = 36
W1[540:600:6, 0] = [0, -4, -8, -12, -16, -20, -24, -28, -32, -36]
b1[0] = -2

# unit 1: left indicator, tanh(-4*(ball_x - paddle_x) - 2)
W1[9:600:60, 1] = -4
W1[15:600:60, 1] = -8
W1[21:600:60, 1] = -12
W1[27:600:60, 1] = -16
W1[33:600:60, 1] = -20
W1[39:600:60, 1] = -24
W1[45:600:60, 1] = -28
W1[51:600:60, 1] = -32
W1[57:600:60, 1] = -36
W1[540:600:6, 1] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
b1[1] = -2

# sharpen the two indicators in the second layer
W2[0, 0] = 10
W2[1, 1] = 10
b2[0] = 0
b2[1] = 0

# decode: action 0=no-op, 1=left, 3=right; suppress 2,4,5
W3[0, 0] = -20
W3[1, 0] = -20
b3[0] = 0
W3[0, 1] = -20
W3[1, 1] = 20
b3[1] = 0
W3[0, 3] = 20
W3[1, 3] = -20
b3[3] = 0
b3[2] = -40
b3[4] = -40
b3[5] = -40
```