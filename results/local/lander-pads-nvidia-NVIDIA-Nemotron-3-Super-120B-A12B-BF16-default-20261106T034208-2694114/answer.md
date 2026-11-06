

```edits
# W1: feature 0 (error_x_norm = -x + 0.2*t0 + 0.5*t1 + 0.8*t2)
W1[0, 0] = -1          # weight for x_norm
W1[5, 0] = 0.2         # t0 weight
W1[6, 0] = 0.5         # t1 weight
W1[7, 0] = 0.8         # t2 weight

# W1: feature 1 (-vy)
W1[3, 1] = -1          # weight for vy

# W1: feature 2 (fuel_norm)
W1[4, 2] = 1.0

# W1: feature 3 (constant 1) will be added via bias b1

# b1: set bias for constant feature, zero others
b1[0:3] = 0
b1[4:16] = 0
b1[3] = 1.0

# W2: scale features by s = 0.2 to keep tanh in linear regime
W2[0, 0] = 0.2
W2[1, 1] = 0.2
W2[2, 2] = 0.2
W2[3, 3] = 0.2

# W3: map second-layer features to logits
# thrust (action 1) from -vy, fuel, bias
W3[1, 1] = 25.0   # from -vy
W3[2, 1] = 2.5    # from fuel_norm
W3[3, 1] = 1.0    # from constant bias

# left (action 2) from error_x_norm (left when error negative)
W3[0, 2] = -10.0  # -10 * error

# right (action 3) from error_x_norm (right when error positive)
W3[0, 3] = 10.0   # 10 * error

# none (action 0) baseline bias
b3[0] = -1.0
b3[1] = 0.0
b3[2] = 0.0
b3[3] = 0.0
```