YOUR TASK: write, from scratch, the weights of a policy network that plays this game as well as possible. There is no training: no data, no loss, no gradients, no fine-tuning; you choose every number yourself. You have no tools: no code execution, no calculator, nothing to run. This is a single response and you will get no feedback. First describe your design briefly (the strategy, and what the hidden units compute), then write the edits block.

THE GAME: Three Pads, a side-view landing game. A craft starts high above flat ground, under gravity. Three landing pads sit on the ground. One of them is the target. The craft must touch down on the target pad, slowly, before it runs out of fuel. The target can change during flight.

The game's exact implementation follows. It is the complete rules. `rng` returns a uniform random number in [0, 1).

```javascript
{{SOURCE}}
```

THE INPUT (8 numbers): the array returned by observe(s), in this order:
  input 0: x / 100
  input 1: y / 60
  input 2: vx
  input 3: vy
  input 4: fuel / 300
  input 5: 1 if the target is pad 0 (x = 20), else 0
  input 6: 1 if the target is pad 1 (x = 50), else 0
  input 7: 1 if the target is pad 2 (x = 80), else 0
THE OUTPUT (4 numbers, logits) for the actions 0 none, 1 thrust, 2 left, 3 right; the action is sampled from the softmax of the logits. With probability 0.1 the game repeats the previous action instead (sticky actions).

THE NETWORK: a multilayer perceptron with 8 inputs, two hidden layers of 16 units with tanh activations, and 4 outputs; 484 parameters:
  h1 = tanh(x @ W1 + b1)          W1: 8 x 16,  b1: 16
  h2 = tanh(h1 @ W2 + b2)         W2: 16 x 16, b2: 16
  logits = h2 @ W3 + b3           W3: 16 x 4,  b3: 4
Row i of W1 holds the weights from input i to the 16 first-layer units; column j of a weight matrix belongs to unit j of the next layer. EVERY WEIGHT STARTS AT 0; you set the weights you need.

HOW YOUR NETWORK IS SCORED: the share of episodes that end with the outcome 'landed', over 3,000 episodes with the three pads used equally, each action sampled from the softmax of the logits, with episode seeds you never see. A second score uses the same count of episodes, but the target changes to another pad once, at a tick between 30 and 120. The first score is the main one.

HOW TO WRITE THE WEIGHTS (numbers only; nothing is run). Write a code block whose opening line is three backticks followed by the word edits, one statement per line ('#' starts a comment; a long list may continue over the next lines). Tensors: W1, b1, W2, b2, W3, b3.

```edits
W1[2, 0] = 2.5                      # one entry: row 2, column 0
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]    # explicit values along a range (as many values as entries)
b1[0:4] = -1                        # fill a range (a:b is half-open; ':' alone = the whole axis)
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]   # a 2-D list for a 2-D block (here 2 x 3)
W3[:, 2] *= 2                       # scale (also += and -=)
```
Ranges: a:b is half-open, a:b:s takes every s-th index.
Statements apply in order; a later statement overwrites an earlier one. Only the LAST edits block of your reply is read; a statement that cannot be read is skipped (the others still apply). Values: plain decimal numbers (|v| <= 1e6).
