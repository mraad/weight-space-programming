

**Design**  
The network implements a simple proportional‑derivative controller.  
- Hidden unit 0 ≈ 0.5·(target x – craft x) – positive when the craft is left of the target.  
- Hidden unit 1 ≈ 0.5·(craft x – target x) – positive when the craft is right of the target.  
- Hidden unit 2 ≈ ‑0.5·(y/HEIGHT) – positive when the craft is low (close to the ground).  

The output layer turns these signals into logits:  
- **left** thruster ∝ +(craft x – target x) (fire left when too far right).  
- **right** thruster ∝ +(target x – craft x) (fire right when too far left).  
- **thrust** ∝ ‑y (fire main engine when low to slow descent).  
- **none** gets a constant negative bias so it is only chosen when no correction is needed.  

All weights are kept small enough that the tanh units operate in their near‑linear region, so the whole network behaves approximately as the linear controller described above.

```edits
# ---- W1 (8×16) -------------------------------------------------
# unit 0: 0.5*(target_x - x)
W1[0,0] = -0.5
W1[5,0] =  0.1   # 0.5*0.2
W1[6,0] =  0.25  # 0.5*0.5
W1[7,0] =  0.4   # 0.5*0.8

# unit 1: 0.5*(x - target_x)
W1[0,1] =  0.5
W1[5,1] = -0.1   # -0.5*0.2
W1[6,1] = -0.25  # -0.5*0.5
W1[7,1] = -0.4   # -0.5*0.8

# unit 2: -0.5*y
W1[1,2] = -0.5

# all other W1 entries remain 0 (implicitly)

# ---- b1 (16) ---------------------------------------------------
b1[:] = 0

# ---- W2 (16×16) ------------------------------------------------
# preserve the first three units (identity)
W2[0,0] = 1
W2[1,1] = 1
W2[2,2] = 1
# all other W2 entries remain 0

# ---- b2 (16) ---------------------------------------------------
b2[:] = 0

# ---- W3 (16×4) ------------------------------------------------
# row 0 -> right (action 3)
W3[0,3] = 4
# row 1 -> left  (action 2)
W3[1,2] = 4
# row 2 -> thrust(action 1)
W3[2,1] = 8
# all other W3 entries remain 0

# ---- b3 (4) ----------------------------------------------------
# action order: none, thrust, left, right
b3[0] = -1   # none
b3[1] = 2    # thrust
b3[2] = -0.5 # left
b3[3] = -0.5 # right
```