

Strategy: first layer builds a constant and crisp step features for vertical speed and for the horizontal control law. The horizontal law uses the signed error to the selected target center, with a conservative braking slope: left is favored when `vx + 0.009*(x - target_x)` is above a small deadband, right when it is below. Second layer combines those features into four action scores with priority: emergency vertical thrust, then horizontal side thrust, then normal descent thrust, then none. The final layer scales these scores into large logits so sampling is near-deterministic.

```edits
# First layer: constant
b1[0] = 10

# First layer: vy emergency feature, positive when vy > -0.2
W1[3, 2] = 100
b1[2] = 20

# First layer: vy normal feature, positive when vy > -0.13
W1[3, 3] = 100
b1[3] = 13

# First layer: left command
# h = vx + 0.009 * (x - target_x); left when h > 0.025
W1[2, 4] = 100
W1[0, 4] = 90
W1[5, 4] = -18
W1[6, 4] = -45
W1[7, 4] = -72
b1[4] = -2.5

# First layer: right command
# right when h < -0.025
W1[2, 5] = -100
W1[0, 5] = -90
W1[5, 5] = 18
W1[6, 5] = 45
W1[7, 5] = 72
b1[5] = -2.5

# Second layer: thrust score
# high when vy is low, unless a side command is active
W2[0, 0] = 1
W2[2, 0] = -1
W2[3, 0] = -1
W2[4, 0] = -0.5
W2[5, 0] = -0.5

# Second layer: left score
# high when left command is active and not in vertical emergency
W2[2, 1] = 2
W2[4, 1] = 2

# Second layer: right score
# high when right command is active and not in vertical emergency
W2[2, 2] = 2
W2[5, 2] = 2

# Second layer: none score
# high only when vy is not low and no side command is active
W2[3, 3] = 1.5
W2[4, 3] = -0.5
W2[5, 3] = -0.5
b2[3] = -0.5

# Output layer: map scores to action logits
# actions: 0 none, 1 thrust, 2 left, 3 right
W3[3, 0] = 50
W3[0, 1] = 50
W3[1, 2] = 50
W3[2, 3] = 50

b3[:] = 0
```