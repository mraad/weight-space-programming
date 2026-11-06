

```edits
# Input to first hidden layer (8 inputs -> 16 units)
W1[0,0] = -1.0          # x -> error_x contribution
W1[5,0] = 0.2           # target pad 0
W1[6,0] = 0.5           # target pad 1
W1[7,0] = 0.8           # target pad 2
W1[3,1] = -1.0          # vy -> -vy
W1[4,2] = 1.0           # fuel -> fuel_norm
b1[3] = 1.0             # constant bias unit

# Second hidden layer (16 -> 16)
W2[0,0] = 1.0           # pass error_x through
W2[1,1] = 1.0           # pass -vy through
b2[3] = 1.0             # constant bias unit

# Output layer (16 -> 4)
W3[0,2] = -2.0          # error_x -> left action
W3[0,3] = 2.0           # error_x -> right action
W3[1,1] = 2.0           # -vy -> thrust action
# all other weights and biases remain zero (defaults)
```
