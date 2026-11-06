

```edits
# Input to first layer (W1) – compute 0.5*horizontal error, 0.5*vy, 0.5*altitude, 0.5*fuel
W1[0,0] = -0.5          # -0.5 * x/100
W1[5,0] = 0.1           # +0.1 * target0
W1[6,0] = 0.25          # +0.25 * target1
W1[7,0] = 0.4           # +0.4 * target2
W1[3,1] = 0.5           # 0.5 * vy
W1[1,2] = 0.5           # 0.5 * y/60 (altitude)
W1[4,3] = 0.5           # 0.5 * fuel/300 (unused)

# Biases for first layer – all zero
# (b1 defaults to zero)

# Second layer – identity (preserve features)
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
# Biases for second layer – all zero
# (b2 defaults to zero)

# Output layer (W3, b3)
b3[0] = 0               # bias for 'none'
b3[1] = 1.0             # bias for 'thrust'
b3[2] = 0               # bias for 'left'
b3[3] = 0               # bias for 'right'

# none action: increases with altitude
W3[2,0] = 2.8           # from altitude unit (index 2)

# thrust action: bias minus altitude term
W3[2,1] = -4.6          # from altitude unit

# left action: negative horizontal error
W3[0,2] = -6.2          # from horizontal error unit (index 0)

# right action: positive horizontal error
W3[0,3] = 6.2           # from horizontal error unit
# All other W3 entries remain zero
```