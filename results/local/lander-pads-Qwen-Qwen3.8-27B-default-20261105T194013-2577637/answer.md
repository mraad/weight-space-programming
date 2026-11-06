

Design brief:
- The policy is a hand-designed PD/bang-bang controller.
- First hidden units 0 and 1 compute a right/left steering score based on position error to the current target and velocity, with a deadband.
- First hidden units 2, 3, and 4 compute vertical-velocity thresholds used to choose thrust vs no-thrust.
- The second hidden layer sharpens these features with a diagonal gain of 2.
- Output logits prioritize thrust when vy is too negative, horizontal thrusts when steering is needed and vertical velocity is not too negative, and otherwise none.

```edits
# right score: 6 * ((5*(target_x - x) - 3*vx) - 0.1)
W1[0,0] = -30.0
W1[2,0] = -18.0
W1[5,0] = 6.0
W1[6,0] = 15.0
W1[7,0] = 30.0
b1[0] = -0.6

# left score: 6 * ((-5*(target_x - x) + 3*vx) - 0.1)
W1[0,1] = 30.0
W1[2,1] = 18.0
W1[5,1] = -6.0
W1[6,1] = -15.0
W1[7,1] = -30.0
b1[1] = -0.6

# vertical velocity gates
W1[3,2] = 6.0
b1[2] = 0.9
W1[3,3] = 6.0
b1[3] = 0.48
W1[3,4] = 6.0
b1[4] = 1.2

# sharpen first-layer features
W2[0,0] = 2.0
W2[1,1] = 2.0
W2[2,2] = 2.0
W2[3,3] = 2.0
W2[4,4] = 2.0
W2[5,5] = 2.0
W2[6,6] = 2.0
W2[7,7] = 2.0
W2[8,8] = 2.0
W2[9,9] = 2.0
W2[10,10] = 2.0
W2[11,11] = 2.0
W2[12,12] = 2.0
W2[13,13] = 2.0
W2[14,14] = 2.0
W2[15,15] = 2.0

# none logit
W3[2,0] = 5.0
W3[3,0] = 5.0
b3[0] = -2.0

# thrust logit
W3[4,1] = -12.0
W3[3,1] = -16.0
b3[1] = -2.0

# left logit
W3[1,2] = 10.0
W3[2,2] = 5.0
b3[2] = -2.0

# right logit
W3[0,3] = 10.0
W3[2,3] = 5.0
b3[3] = -2.0
```