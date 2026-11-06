

```edits
# Identity mapping for the first hidden layer (copy inputs)
W1[0,0] = 1
W1[1,1] = 1
W1[2,2] = 1
W1[3,3] = 1
W1[4,4] = 1
W1[5,5] = 1
W1[6,6] = 1
W1[7,7] = 1
b1[0:16] = 0

# Identity mapping for the second hidden layer (copy first 8 units)
W2[0,0] = 1
W2[1,1] = 1
W2[2,2] = 1
W2[3,3] = 1
W2[4,4] = 1
W2[5,5] = 1
W2[6,6] = 1
W2[7,7] = 1
b2[0:16] = 0

# Output layer: compute logits for none, thrust, left, right
# Row 0: x_norm
W3[0,0] = 0   # none
W3[0,1] = 0   # thrust
W3[0,2] = 4.0 # left
W3[0,3] = -4.0# right
# Row 1: y_norm
W3[1,0] = 0
W3[1,1] = 2.0
W3[1,2] = 0
W3[1,3] = 0
# Row 2: vx
W3[2,0] = 0
W3[2,1] = 0
W3[2,2] = 2.0
W3[2,3] = -2.0
# Row 3: vy
W3[3,0] = 0
W3[3,1] = -2.0
W3[3,2] = 0
W3[3,3] = 0
# Row 4: fuel (unused)
W3[4,0] = 0
W3[4,1] = 0
W3[4,2] = 0
W3[4,3] = 0
# Row 5: target pad 0 one‑hot
W3[5,0] = 0
W3[5,1] = 0
W3[5,2] = -0.8
W3[5,3] = 0.8
# Row 6: target pad 1 one‑hot
W3[6,0] = 0
W3[6,1] = 0
W3[6,2] = -2.0
W3[6,3] = 2.0
# Row 7: target pad 2 one‑hot
W3[7,0] = 0
W3[7,1] = 0
W3[7,2] = -3.2
W3[7,3] = 3.2
# Rows 8‑15 remain zero (no contribution)
W3[8:16, :] = 0

# Biases for logits (no offset)
b3[0:4] = 0
```