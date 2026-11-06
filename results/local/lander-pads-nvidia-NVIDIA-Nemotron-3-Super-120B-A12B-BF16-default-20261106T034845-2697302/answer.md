

**Strategy**  
The network is kept in the quasi‑linear regime of the tanh activations by using very small weights in the first layer (scale = 0.1).  
The first hidden layer computes five useful features:  

* h₀ ≈ 0.1·(x – targetₓ)  horizontal error to the target pad  
* h₁ ≈ 0.1·y  height above ground  
* h₂ ≈ 0.1·vₓ  horizontal velocity  
* h₃ ≈ 0.1·v_y  vertical velocity  
* h₄ ≈ 0.1·fuel  remaining fuel  

A second identity layer preserves these features (still in the linear tanh region).  
The output layer forms logits for the four actions:

* **Thrust (1)** – fires when the craft is low (small h₁), falling fast (negative h₃) and has fuel (positive h₄).  
* **Left (2)** – fires when the craft is right of the target (positive h₀) or moving right (positive h₂).  
* **Right (3)** – fires when the craft is left of the target (negative h₀) or moving left (negative h₂).  
* **None (0)** – baseline zero logit.

Biases are set so that each thruster only activates when its feature‑based term overcomes a small negative threshold, preventing unnecessary side‑thrust when the craft is already centered and still.

---

```edits
# Zero all parameters first
W1[:,:] = 0
b1[:] = 0
W2[:,:] = 0
b2[:] = 0
W3[:,:] = 0
b3[:] = 0

# First layer: compute features (scale = 0.1)
W1[0,0] = 0.1          # x
W1[0,5] = -0.02        # -0.2 * target0
W1[0,6] = -0.05        # -0.5 * target1
W1[0,7] = -0.08        # -0.8 * target2
W1[1,1] = 0.1          # y
W1[2,2] = 0.1          # vx
W1[3,3] = 0.1          # vy
W1[4,4] = 0.1          # fuel

# Second layer: identity (preserve features)
W2[0,0] = 1.0
W2[1,1] = 1.0
W2[2,2] = 1.0
W2[3,3] = 1.0
W2[4,4] = 1.0
W2[5,5] = 1.0
W2[6,6] = 1.0
W2[7,7] = 1.0
W2[8,8] = 1.0
W2[9,9] = 1.0
W2[10,10] = 1.0
W2[11,11] = 1.0
W2[12,12] = 1.0
W2[13,13] = 1.0
W2[14,14] = 1.0
W2[15,15] = 1.0

# Output layer (logits)
# Thrust (action 1)
W3[1,1] = -20.0   # h1 (height) – lower height → more thrust
W3[3,1] = -5.0    # h3 (vertical velocity) – falling → more thrust
W3[4,1] = 20.0    # h4 (fuel) – more fuel → more thrust
# Left (action 2)
W3[0,2] = 8.0     # h0 (horizontal error) – right of target → left thrust
W3[2,2] = 3.0     # h2 (horizontal velocity) – moving right → left thrust
# Right (action 3)
W3[0,3] = -8.0    # h0 – left of target → right thrust
W3[2,3] = -3.0    # h2 – moving left → right thrust
# Biases
b3[0] = 0.0       # none
b3[1] = -1.0      # thrust bias
b3[2] = -0.05     # left bias
b3[3] = -0.05     # right bias
```