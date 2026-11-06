

Design:

- Use the first 10 hidden units as binary detectors for the ball’s column (channel 3).  
- Use hidden units 10–19 as binary detectors for the paddle’s column (channel 0).  
- Hidden unit 20 computes the horizontal error:  
  \[
  e = x_\text{ball} - x_\text{paddle}
  \]
  using the column detectors.
- The output layer makes left/right logits an antisymmetric function of \(e\), with no-op winning when \(e=0\). Up/down/fire are heavily suppressed.
- This policy simply steers the paddle toward the ball’s current column. That keeps the paddle under the ball when it reaches the bottom row and avoids relying on the ambiguous trail after side paddles.

```edits
# Ball column detectors: units 0..9
W1[3:600:60, 0] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
W1[9:600:60, 1] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
W1[15:600:60, 2] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
W1[21:600:60, 3] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
W1[27:600:60, 4] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
W1[33:600:60, 5] = [10, 10, 10, 10,