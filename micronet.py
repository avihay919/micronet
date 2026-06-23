"""
micronet.py  —  a neural network FROM SCRATCH (NumPy only).

This is a scaffold. The structure, the weight init, the numerically-stable
softmax, and the training loop are written for you (boilerplate). The parts
that matter and that an interviewer WILL ask you to explain — the forward pass
and backpropagation — are left as TODOs, with the exact formulas you need in
the comments. Implement those yourself. Understand every line.

Architecture: a multi-layer perceptron (MLP)
    input -> [Linear -> ReLU] x (hidden layers) -> Linear -> softmax
    loss: cross-entropy

Run:  python3 micronet.py
Goal: >95% test accuracy on the two-moons data once forward+backward are done.
"""

import numpy as np
from data import make_moons, one_hot, train_test_split


# ----------------------------- helpers (given) -----------------------------

def softmax(z):
    """Numerically stable softmax over the last axis. (Given — it's fiddly.)"""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_onehot):
    """Mean cross-entropy loss. (Given.)"""
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

def relu(z):
    return np.maximum(0, z)


# ------------------------------- the model ---------------------------------

class MLP:
    def __init__(self, sizes, seed=0):
        """sizes e.g. [2, 16, 16, 2] -> 2 inputs, two hidden layers of 16, 2 classes."""
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            # He initialisation (good for ReLU). Given — it's standard boilerplate.
            self.W.append(rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros((1, sizes[i + 1])))

    def forward(self, X):
        a = X
        zs, activations = [], [X]
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            if i < L - 1:
                a = relu(z)
            else:
                a = softmax(z)
            activations.append(a)
        self.cache = {"z": zs, "a": activations}
        return a
    
    def backward(self, X, y_onehot, lr):
        probs = self.cache["a"][-1]
        N = X.shape[0]
        dz = (probs - y_onehot) / N

        L = len(self.W)
        for i in reversed(range(L)):
            a_prev = self.cache["a"][i]          # activation feeding INTO layer i
            dW = a_prev.T @ dz
            db = dz.sum(axis=0, keepdims=True)

            if i > 0:
                da_prev = dz @ self.W[i].T
                z_prev = self.cache["z"][i - 1]
                dz = da_prev * (z_prev > 0)      # set up dz for the next loop iteration

            self.W[i] -= lr * dW
            self.b[i] -= lr * db
    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def accuracy(model, X, y):
    return (model.predict(X) == y).mean()


# ------------------------------ training (given) ----------------------------

def train(sizes=(2, 16, 16, 2), epochs=300, lr=0.5, seed=0):
    X, y = make_moons(seed=seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, seed=seed)
    Ytr = one_hot(ytr)

    model = MLP(list(sizes), seed=seed)
    for ep in range(epochs):
        probs = model.forward(Xtr)
        loss = cross_entropy(probs, Ytr)
        model.backward(Xtr, Ytr, lr)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  loss {loss:.4f}  "
                  f"train acc {accuracy(model, Xtr, ytr):.3f}  "
                  f"test acc {accuracy(model, Xte, yte):.3f}")
    return model


if __name__ == "__main__":
    train()
