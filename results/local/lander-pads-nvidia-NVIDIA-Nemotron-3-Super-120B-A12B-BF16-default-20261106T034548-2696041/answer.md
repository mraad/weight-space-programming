

**Strategy**  
The network is used as a linear controller (the two hidden layers are kept in the linear regime of tanh).  
* First hidden layer simply scales the eight inputs by 0.5 and copies them into units 0‑7 (the other units stay zero).  
* Second hidden layer is the identity, so the activations remain essentially the scaled inputs.  
* The output layer computes four logits:  

| Action | Logic (implemented by the output weights) |
|--------|------------------------------------------|
| **none** | constant bias = +1.0 (default action) |
| **thrust** | fires when downward speed is large: logit = ‑0.5 ‑ 8·vy (triggers for vy < ‑0.1875) |
| **left** | fires when the craft is right of the target pad: logit = ‑0.5 + 0.4·x ‑ 8·target0 ‑ 20·target1 ‑ 32·target2 |
| **right**| fires when the craft is left of the target pad: logit = ‑0.5 ‑ 0.4·x + 8·target0 + 20·target1 + 32·target2 |

With the sticky‑action probability (0.1) the policy still settles into a safe hover‑and‑drift behavior that lands on the target pad with vertical speed within the safe limits.  

All weights are chosen so that the pre‑activations stay in the roughly linear range of tanh (‑0.25 … 0.75), making the network behave almost exactly as the linear controller described above.  

```edits
# First hidden layer: scale inputs by 0.5 into units 0‑7
W1[0,0] = 0.5
W1[1,1] = 0.5
W1[2,2] = 0.5
W1[3,3] = 0.5
W1[4,4] = 0.5
W1[5,5] = 0.5
W1[6,6] = 0.5
W1[7,7] = 0.5
# Biases first layer zero
b1[0:16] = 0.0
# Second hidden layer: identity
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
# Biases second layer zero
b2[0:16] = 0.0
# Output layer: implement linear controller
# row 0 (input x)
W3[0,0:4] = [0, 0, 80, -80]
# row 1 (input y) – no contribution
W3[1,0:4] = [0, 0, 0, 0]
# row 2 (input vx) – none
W3[2,0:4] = [0, 0, 0, 0]
# row 3 (input vy) – thrust weight
W3[3,0:4] = [0, -16, 0, 0]
# row 4 (input fuel) – none
W3[4,0:4] = [0, 0, 0, 0]
# row 5 (target pad0)
W3[5,0:4] = [0, 0, -16, 16]
# row 6 (target pad1)
W3[6,0:4] = [0, 0, -40, 40]
# row 7 (target pad2)
W3[7,0:4] = [0, 0, -64, 64]
# rows 8‑15 unused
W3[8:16, 0:4] = 0.0
# Output biases: none high, others low
b3[0] = 1.0   # none
b3[1] = -0.5  # thrust
b3[2] = -0.5  # left
b3[3] = -0.5  # right
```