

Strategy:
- Let `e = x/100 - (0.2*target0 + 0.5*target1 + 0.8*target2)` be the normalized horizontal error from the current target.
- First hidden units compute saturated bang-bang signals:
  - left when `vx + 4e` is sufficiently positive,
  - right when `vx + 4e` is sufficiently negative,
  - thrust when `vy` is below about `-0.15`.
- Second hidden units amplify those three signals into near-binary desire values.
- Output logits strongly prefer the desired action. Vertical thrust has the largest gain, so if the craft is falling too fast it temporarily overrides horizontal control. Otherwise the side controller drives the craft to the target and kills velocity.

```edits
# h1[0]: left  tanh(50 * (vx + 4e - 0.08))
# h1[1]: right tanh(50 * (-vx - 4e - 0.08))
# h1[2]: thrust tanh(50 * (-vy - 0.15))
W1[0, 0] = 200
W1[0, 1] = -200
W1[2, 0] = 50
W1[2, 1] = -50
W1[3, 2] = -50
W1[5, 0] = -40
W1[5, 1] = 40
W1[6, 0] = -100
W1[6, 1] = 100
W1[7, 0] = -160
W1[7, 1] = 160
b1[0] = -4
b1[1] = -4
b1[2] = -7.5
# h2[0]: thrust desire from h1[2]
# h2[1]: left desire from h1[0]
# h2[2]: right desire from h1[1]
W2[2, 0] = 5
W2[0, 1] = 5
W2[1, 2] = 5
# Outputs: none=0, thrust=1, left=2, right=3
W3[0, 1] = 50
W3[1, 2] = 30
W3[2, 3] = 30
b3[0] = 10
b3[1] = 0
b3[2] = -10
b3[3] = -10
```