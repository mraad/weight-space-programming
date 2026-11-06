

I use a hand-crafted controller, not trained features.

- Hidden layer 1 computes three saturated gates:
  - right gate: `150 * ((target_x/100) - (x/100) - vx) - 1.5`
  - left gate: `150 * (-(target_x/100) + (x/100) + vx) - 1.5`
  - thrust gate: `-10*vy - 4*(y/60) - 0.5`
  This makes horizontal control a PD/sliding-mode approach to the target pad, and vertical control a skydive/braking law that keeps touchdown speed safe.
- Hidden layer 2 converts the gates into action activations and a “none” activation:
  - right activation: `tanh(3*right_gate - 1)`
  - left activation: `tanh(3*left_gate - 1)`
  - thrust activation: `tanh(3*thrust_gate - 0.5)`
  - none activation: `tanh(-2.5 - 1.7*(right_gate + left_gate + thrust_gate))`
- Output logits:
  - none = `8*none_act + 2`
  - thrust = `8*thrust_act`
  - left = `8*left_act - 2*thrust_act`
  - right = `8*right_act - 2*thrust_act`
  This gives thrust priority when vertical speed is too high, side thrust when horizontal error remains, and none otherwise.

```edits
W1[0, 0] = -150.0
W1[2, 0] = -150.0
W1[5, 0] = 30.0
W1[6, 0] = 75.0
W1[7, 0] = 120.0
b1[0] = -1.5
W1[0, 1] = 150.0
W1[2, 1] = 150.0
W1[5, 1] = -30.0
W1[6, 1] = -75.0
W1[7, 1] = -120.0
b1[1] = -1.5
W1[1, 2] = -4.0
W1[3, 2] = -10.0
b1[2] = -0.5
W2[0, 0] = 3.0
b2[0] = -1.0
W2[1, 1] = 3.0
b2[1] = -1.0
W2[2, 2] = 3.0
b2[2] = -0.5
W2[3, 0] = -1.7
W2[3, 1] = -1.7
W2[3, 2] = -1.7
b2[3] = -2.5
W3[0, 3] = 8.0
W3[2, 3] = -2.0
W3[1, 2] = 8.0
W3[2, 2] = -2.0
W3[2, 1] = 8.0
W3[3, 0] = 8.0
b3[0] = 2.0
```