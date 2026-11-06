

```edits
# Hidden unit 0: -vy (vertical speed downward)
W1[3,0] = -1
b1[0] = 0

# Hidden unit 1: horizontal error = target_x_norm - x_norm
W1[0,1] = -1               # subtract x_norm
W1[5,1] = 0.2              # +0.2 if target pad 0
W1[6,1] = 0.5              # +0.5 if target pad 1
W1[7,1] = 0.8              # +0.8 if target pad 2
b1[1] = 0

# Hidden layer 2: identity (preserve h1 values)
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
b2[0:16] = 0

# Output layer: map hidden units to action logits
# thrust (action 1) from unit0 (-vy)
W3[0,1] = 4
# left (action 2) from unit1 (error_x) -> need negative error
W3[1,2] = -4
# right (action 3) from unit1 (error_x) -> need positive error
W3[1,3] = 4
# bias for none (action 0)
b3[0] = 0.5
# all other biases and weights remain zero (default)
```
