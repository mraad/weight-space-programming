

Design: I made the network implement a hand-written Breakout controller.

- First hidden layer extracts simple thresholded features:
  - unit 0: whether the ball is moving downward (ball row below trail row; same row is treated as “not down”).
  - units 1/2: horizontal ball direction from ball column vs. trail column.
  - units 3/4: whether the paddle is at or to the right/left of the ball, with equality counted as true.
  - units 5/6: whether the paddle is at least two cells to the right of the ball, or the ball is at least two cells to the right of the paddle.
  - units 7/8: whether the paddle is right/left of center column 4.
- Second hidden layer forms thresholded AND conditions:
  - If the ball is falling and moving right, move right when the paddle is at/left of the ball, and move left when the paddle is more than one cell to the right.
  - If the ball is falling and moving left, move left when the paddle is at/right of the ball, and move right when the paddle is more than one cell to the left.
  - If the ball is not falling, move the paddle toward center column 4.
- Output layer strongly activates left/right for those conditions, leaves no-op high when no condition is active, and suppresses up/down/fire.

All unspecified weights remain 0.

```edits
# First-layer biases
b1[0:9] = [-3, 0, 0, 2.5, 2.5, -7.5, -7.5, -16, 16]

# h1 unit 0: down = ball_row - trail_row (same row biased to -1)
W1[63:120:6, 0] = 6
W1[64:120:6, 0] = -6
W1[123:180:6, 0] = 12
W1[124:180:6, 0] = -12
W1[183:240:6, 0] = 18
W1[184:240:6, 0] = -18
W1[243:300:6, 0] = 24
W1[244:300:6, 0] = -24
W1[303:360:6, 0] = 30
W1[304:360:6, 0] = -30
W1[363:420:6, 0] = 36
W1[364:420:6, 0] = -36
W1[423:480:6, 0] = 42
W1[424:480:6, 0] = -42
W1[483:540:6, 0] = 48
W1[484:540:6, 0] = -48
W1[543:600:6, 0] = 54
W1[544:600:6, 0] = -54

# h1 unit 1: right_dir = ball_col - trail_col
W1[9:600:60, 1] = 5
W1[10:600:60, 1] = -5
W1[15:600:60, 1] = 10
W1[16:600:60, 1] = -10
W1[21:600:60, 1] = 15
W1[22:600:60, 1] = -15
W1[27:600:60, 1] = 20
W1[28:600:60, 1] = -20
W1[33:600:60, 1] = 25
W1[34:600:60, 1] = -25
W1[39:600:60, 1] = 30
W1[40:600:60, 1] = -30
W1[45:600:60, 1] = 35
W1[46:600:60, 1] = -35
W1[51:600:60, 1] = 40
W1[52:600:60, 1] = -40
W1[57:600:60, 1] = 45
W1[58:600:60, 1] = -45

# h1 unit 2: left_dir = trail_col - ball_col
W1[9:600:60, 2] = -5
W1[10:600:60, 2] = 5
W1[15:600:60, 2] = -10
W1[16:600:60, 2] = 10
W1[21:600:60, 2] = -15
W1[22:600:60, 2] = 15
W1[27:600:60, 2] = -20
W1[28:600:60, 2] = 20
W1[33:600:60, 2] = -25
W1[34:600:60, 2] = 25
W1[39:600:60, 2] = -30
W1[40:600:60, 2] = 30
W1[45:600:60, 2] = -35
W1[46:600:60, 2] = 35
W1[51:600:60, 2] = -40
W1[52:600:60, 2] = 40
W1[57:600:60, 2] = -45
W1[58:600:60, 2] = 45

# h1 unit 3: p_le_x = ball_col - paddle_col (true when ball is not left of paddle)
W1[9:600:60, 3] = 5
W1[15:600:60, 3] = 10
W1[21:600:60, 3] = 15
W1[27:600:60, 3] = 20
W1[33:600:60, 3] = 25
W1[39:600:60, 3] = 30
W1[45:600:60, 3] = 35
W1[51:600:60, 3] = 40
W1[57:600:60, 3] = 45
W1[540:600:6, 3] = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45]

# h1 unit 4: p_ge_x = paddle_col - ball_col (true when paddle is not left of ball)
W1[9:600:60, 4] = -5
W1[15:600:60, 4] = -10
W1[21:600:60, 4] = -15
W1[27:600:60, 4] = -20
W1[33:600:60, 4] = -25
W1[39:600:60, 4] = -30
W1[45:600:60, 4] = -35
W1[51:600:60, 4] = -40
W1[57:600:60, 4] = -45
W1[540:600:6, 4]