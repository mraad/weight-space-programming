

The policy is a hand-built controller, not a learned approximation:

- **Horizontal control:**  
  Compute a damped horizontal error  
  `e = target_x_norm - x_norm - 0.6 * vx`.  
  Positive `e` means “target is to the right / need right thrust”; negative means “need left thrust”.  
  The velocity term provides braking/overshoot damping. The controller uses a deadband of about 2 game units (`0.02` normalized).

- **Vertical control:**  
  Use a bang-bang rule on vertical velocity. Thrust when `vy` is below about `-0.15`; otherwise allow horizontal/none. This keeps the descent speed safely below the crash threshold while preserving enough side-thrust duty cycle.

- **Hidden units:**
  - `h1[0] = tanh(target_x_norm - x_norm - 0.6*vx)`  
  - `h1[1] = tanh(5*vy)`  
  - `h1[2] = tanh(y_norm)`  
  - `h2[0]` = saturated right-command detector  
  - `h2[1]` = saturated left-command detector  
  - `h2[2]` = saturated thrust-command detector  
  - `h2[3]` = low-height detector

- **Output logits:**  
  None is the default. Thrust dominates when the vertical thrust detector is active. Left/right dominate when their horizontal detector is active and thrust is not active. Low height gives a small side-thrust boost to help final alignment without overriding vertical safety.

```edits
# h1[0]: damped horizontal error: target_x_norm - x_norm - 0.6*vx
W1[0, 0] = -1.0
W1[2, 0] = -0.6
W1[5, 0] = 0.2
W1[6, 0] = 0.5
W1[7, 0] = 0.8

# h1[1]: scaled vy
W1[3, 1] = 5.0

# h1[2]: scaled height
W1[1, 2] = 1.0

# h2[0]: right when horizontal error > 0.02
W2[0, 0] = 200.0
b2[0] = -4.0

# h2[1]: left when horizontal error < -0.02
