# MicroNet — a neural network from scratch in NumPy

A small multi-layer perceptron implemented from first principles — forward pass
and backpropagation written by hand, with no deep-learning frameworks. It trains
to ~98.5% test accuracy on a non-linearly-separable classification task.

The point of the project is to understand what a neural network actually does
under the hood: the linear layers, the non-linearity, the loss, and especially
the gradient computation that drives learning — rather than calling a library
that hides all of it.

## What it does

- Generates a synthetic "two moons" dataset — two interleaving classes that no
  straight line can separate, so solving it requires real hidden layers and a
  non-linearity.
- Implements a configurable MLP: linear layers, ReLU activations, a softmax
  output, and cross-entropy loss.
- Implements backpropagation by hand — gradients derived and computed from
  scratch, with stochastic gradient-descent weight updates.
- Trains the network and reports loss and train/test accuracy each epoch,
  reaching ~98.5% test accuracy.

## Architecture

```
input (2-D point)
  -> Linear -> ReLU       (hidden layer 1)
  -> Linear -> ReLU       (hidden layer 2)
  -> Linear -> Softmax    (output: class probabilities)
loss: cross-entropy
```

The network shape is configurable via a single list, e.g. `[2, 16, 16, 2]`
(2 inputs, two hidden layers of 16 units, 2 output classes).

## Run

```bash
pip install numpy matplotlib
python data.py        # generate + visualise the dataset (writes data.png)
python micronet.py    # train the network and print loss / accuracy per epoch
```

## Files

- `micronet.py` — the model: weight initialisation, forward pass, backpropagation,
  and the training loop.
- `data.py` — synthetic two-moons dataset generator and train/test split.
- `data.png` — a visualisation of the dataset.

## Implementation notes

- **No frameworks.** Only NumPy is used for the network itself; there is no
  autograd. Every gradient is computed explicitly in `backward`.
- **Softmax + cross-entropy.** Pairing these means the gradient at the output
  layer simplifies cleanly to `probs - y`, which is where backpropagation
  begins before the error is propagated back through each layer.
- **ReLU** provides the non-linearity that lets the network learn a curved
  decision boundary rather than a single straight cut.