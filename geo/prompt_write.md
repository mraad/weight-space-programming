YOUR TASK: write, from scratch, the weights of a convolutional network that finds circular storage tanks in an overhead RGB image. There is no training: no data, no loss, no gradients, no fine-tuning; you choose every number yourself. You have no tools: no code execution, no calculator, nothing to run. This is a single response and you will get no feedback. First describe your design briefly (what each path of the network computes), then write the edits block.

THE IMAGE: one RGB image of an industrial area, 3221 x 4749 pixels. A pixel is about 0.8 m. Tanks are pale discs seen from above. The image also holds roads, bare ground, buildings and pipe racks.

THE INPUT: the whole image at once, with values pixel / 255, so each value is in [0, 1]. Band order is red, green, blue. Before the forward pass, the image is padded on the bottom and right by repeating its edge, so that 16 divides each side; the padding is cut from the output.

THE NETWORK: the complete source follows. It has 16,961 weights.

```python
{{SOURCE}}
```

A pixel is called a tank if its output logit > 0. The threshold is fixed.

WHAT WAS MEASURED: these numbers come from a train area of the image only. The rest of the image is hidden from you.

{{FACTS}}

HOW YOUR NETWORK IS SCORED: on hidden areas of the same image. The scores are the tank pixel IoU, and the object F1: connected tank regions of at least 50 pixels, matched to true tanks at IoU >= 0.5.

HOW TO WRITE THE WEIGHTS (numbers only; nothing is run). EVERY WEIGHT STARTS AT 0, including every bias; you set the weights you need. A convolution's weight tensor has the shape [out channels, in channels, kernel height, kernel width], so index [o, i, 1, 1] is the centre of the 3 x 3 kernel from input channel i to output channel o. Tensors are named <layer>.weight and <layer>.bias, for example e0a.weight, d3.bias, out.weight. Write a code block whose opening line is three backticks followed by the word edits, one statement per line ('#' starts a comment; a long list may continue over the next lines).

```edits
e0a.weight[0, 0, 1, 1] = 2.5                     # one entry
e0a.weight[0, 0:3, 1, 1] = [1, 0, -1]            # explicit values along a range (as many values as entries)
e1a.weight[1, 0, 0:3, 0:3] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]   # a 2-D list for a 2-D block
d0.bias[0:4] = -1                                # fill a range
out.weight[0, :, 0, 0] *= 2                      # scale (also += and -=)
```
Ranges: a:b is half-open, a:b:s takes every s-th index, ':' alone is the whole axis.
Statements apply in order; a later statement overwrites an earlier one. Only the LAST edits block of your reply is read; a statement that cannot be read is skipped (the others still apply). Values: plain decimal numbers (|v| <= 1e6).
