## Design

I worked this out by hand and haven't run it, so the design and any score expectations are untested.

**Strategy:** the paddle predicts the column where the ball will reach row 8, moves there early, and waits with no-ops. Waiting matters because a sticky repeat of a no-op is harmless, so sticky actions only cost anything when the paddle changes direction.

Aiming directly under the ball also keeps its horizontal direction unchanged at the catch. This matters because horizontal motion (period 20) and vertical motion (period 18) then keep drifting relative to each other. The ball eventually visits every cell, so it shouldn't get stuck in a loop that misses the remaining bricks.

**Landing prediction:** I use `fold(x + dx·S)`, where `fold` applies the game's clamped wall reflection (`u<0 → -1-u`, `u>9 → 19-u`).
- **Ball going down:** `S = 8 - y`, which is exact unless the ball hits a brick from above.
- **Ball going up:** the bounce row isn't known yet, so I assume a bounce around row 3, which gives `S = y + 3` for rows 4–8. The values for rows 0–3 are 9, 10, 9 and 8. The estimate is corrected once the ball starts down.

**First hidden layer:**
- **Units 0–3 compare the predicted column with the paddle.** Each computes `tanh(4·(T − P))`, with one lookup table per direction (right/left, down/up). The output is +1 if the paddle should go right, −1 if it should go left, and 0 if it is already in place.
- **Unit 4 is +1 when the ball moves right.** It uses ball x minus trail x, and also handles the steps where the ball clamps at a wall, including the reset state.
- **Unit 5 is +1 when the ball moves down.** It uses ball y minus trail y. Bounces at the top or off a brick count as down, and a paddle bounce (ball and trail both in row 8) counts as up.

**Second hidden layer:** units 0–7 each AND one comparison (or its negation) with the two direction gates, using `8·(±c ± gx ± gy) − 20`. This gives one "go right" and one "go left" unit for each of the 4 directions.

**Output:** the right logit is +15 if any "go right" unit is active and −15 otherwise, and left works the same way. No-op stays at 0, and up, down and fire are pushed to −30.

```edits
# ---- Unit 0: dx=+1, going down: 4*fold(x+(8-y)) on ball cells
W1[3:60:6, 0] = [32,36,36,32,28,24,20,16,12,8]
W1[63:120:6, 0] = [28,32,36,36,32,28,24,20,16,12]
W1[123:180:6, 0] = [24,28,32,36,36,32,28,24,20,16]
W1[183:240:6, 0] = [20,24,28,32,36,36,32,28,24,20]
W1[243:300:6, 0] = [16,20,24,28,32,36,36,32,28,24]
W1[303:360:6, 0] = [12,16,20,24,28,32,36,36,32,28]
W1[363:420:6, 0] = [8,12,16,20,24,28,32,36,36,32]
W1[423:480:6, 0] = [4,8,12,16,20,24,28,32,36,36]
W1[483:540:6, 0] = [0,4,8,12,16,20,24,28,32,36]
# ---- Unit 1: dx=-1, going down: 4*fold(x-(8-y))
W1[3:60:6, 1] = [28,24,20,16,12,8,4,0,0,4]
W1[63:120:6, 1] = [24,20,16,12,8,4,0,0,4,8]
W1[123:180:6, 1] = [20,16,12,8,4,0,0,4,8,12]
W1[183:240:6, 1] = [16,12,8,4,0,0,4,8,12,16]
W1[243:300:6, 1] = [12,8,4,0,0,4,8,12,16,20]
W1[303:360:6, 1] = [8,4,0,0,4,8,12,16,20,24]
W1[363:420:6, 1] = [4,0,0,4,8,12,16,20,24,28]
W1[423:480:6, 1] = [0,0,4,8,12,16,20,24,28,32]
W1[483:540:6, 1] = [0,4,8,12,16,20,24,28,32,36]
# ---- Unit 2: dx=+1, going up: 4*fold(x+S(y)), S = 9,10,9,8,7,8,9,10,11 for rows 0..8
W1[3:60:6, 2] = [36,36,32,28,24,20,16,12,8,4]
W1[63:120:6, 2] = [36,32,28,24,20,16,12,8,4,0]
W1[123:180:6, 2] = [36,36,32,28,24,20,16,12,8,4]
W1[183:240:6, 2] = [32,36,36,32,28,24,20,16,12,8]
W1[243:300:6, 2] = [28,32,36,36,32,28,24,20,16,12]
W1[303:360:6, 2] = [32,36,36,32,28,24,20,16,12,8]
W1[363:420:6, 2] = [36,36,32,28,24,20,16,12,8,4]
W1[423:480:6, 2] = [36,32,28,24,20,16,12,8,4,0]
W1[483:540:6, 2] = [32,28,24,20,16,12,8,4,0,0]
# ---- Unit 3: dx=-1, going up: 4*fold(x-S(y))
W1[3:60:6, 3] = [32,28,24,20,16,12,8,4,0,0]
W1[63:120:6, 3] = [36,32,28,24,20,16,12,8,4,0]
W1[123:180:6, 3] = [32,28,24,20,16,12,8,4,0,0]
W1[183:240:6, 3] = [28,24,20,16,12,8,4,0,0,4]
W1[243:300:6, 3] = [24,20,16,12,8,4,0,0,4,8]
W1[303:360:6, 3] = [28,24,20,16,12,8,4,0,0,4]
W1[363:420:6, 3] = [32,28,24,20,16,12,8,4,0,0]
W1[423:480:6, 3] = [36,32,28,24,20,16,12,8,4,0]
W1[483:540:6, 3] = [36,36,32,28,24,20,16,12,8,4]
# ---- Paddle column subtracted in units 0-3
W1[540:600:6, 0] = [0,-4,-8,-12,-16,-20,-24,-28,-32,-36]
W1[540:600:6, 1] = [0,-4,-8,-12,-16,-20,-24,-28,-32,-36]
W1[540:600:6, 2] = [0,-4,-8,-12,-16,-20,-24,-28,-32,-36]
W1[540:600:6, 3] = [0,-4,-8,-12,-16,-20,-24,-28,-32,-36]
# ---- Unit 4: gate "ball moving right" (ball columns, then trail columns)
W1[3:600:60, 4] = 8
W1[9:600:60, 4] = 8
W1[15:600:60, 4] = 16
W1[21:600:60, 4] = 24
W1[27:600:60, 4] = 32
W1[33:600:60, 4] = 40
W1[39:600:60, 4] = 48
W1[45:600:60, 4] = 56
W1[51:600:60, 4] = 64
W1[57:600:60, 4] = 72
W1[10:600:60, 4] = -8
W1[16:600:60, 4] = -16
W1[22:600:60, 4] = -24
W1[28:600:60, 4] = -32
W1[34:600:60, 4] = -40
W1[40:600:60, 4] = -48
W1[46:600:60, 4] = -56
W1[52:600:60, 4] = -64
W1[58:600:60, 4] = -72
b1[4] = -4
# ---- Unit 5: gate "ball moving down" (ball rows, then trail rows)
W1[63:120:6, 5] = 8
W1[123:180:6, 5] = 16
W1[183:240:6, 5] = 24
W1[243:300:6, 5] = 32
W1[303:360:6, 5] = 40
W1[363:420:6, 5] = 48
W1[423:480:6, 5] = 56
W1[483:540:6, 5] = 56
W1[64:120:6, 5] = -8
W1[124:180:6, 5] = -16
W1[184:240:6, 5] = -24
W1[244:300:6, 5] = -32
W1[304:360:6, 5] = -40
W1[364:420:6, 5] = -48
W1[424:480:6, 5] = -56
W1[484:540:6, 5] = -64
b1[5] = 4
# ---- Layer 2: AND of comparison with direction gates
# h2 units: 0 R(d+,down) 1 L(d+,down) 2 R(d-,down) 3 L(d-,down) 4 R(d+,up) 5 L(d+,up) 6 R(d-,up) 7 L(d-,up)
W2[0, 0] = 8
W2[0, 1] = -8
W2[1, 2] = 8
W2[1, 3] = -8
W2[2, 4] = 8
W2[2, 5] = -8
W2[3, 6] = 8
W2[3, 7] = -8
W2[4, 0:8] = [8,8,-8,-8,8,8,-8,-8]
W2[5, 0:8] = [8,8,8,8,-8,-8,-8,-8]
b2[0:8] = -20
# ---- Output: right = +15 if any R unit is active, else -15; same for left; no-op = 0
W3[0:8, 3] = [15,0,15,0,15,0,15,0]
W3[0:8, 1] = [0,15,0,15,0,15,0,15]
b3[1] = 45
b3[3] = 45
b3[2] = -30
b3[4] = -30
b3[5] = -30
```