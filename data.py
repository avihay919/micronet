"""
data.py  —  synthetic dataset generator.

Generates a "two moons" dataset: two interleaving half-circles that are NOT
linearly separable. That's deliberate — a plain linear model (no hidden layer)
*cannot* solve this, so getting good accuracy proves your hidden layers and
backprop actually work. No downloads, no external data, fully reproducible.

This file is complete and working. Run it first to see your data.
"""

import numpy as np


def make_moons(n_samples=1000, noise=0.15, seed=0):
    """Return X (n,2) float features and y (n,) int labels in {0,1}."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    # outer moon (class 0)
    theta_out = np.linspace(0, np.pi, n_out)
    x_out = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1)

    # inner moon (class 1), shifted to interleave with the outer one
    theta_in = np.linspace(0, np.pi, n_in)
    x_in = np.stack([1 - np.cos(theta_in), 0.5 - np.sin(theta_in)], axis=1)

    X = np.vstack([x_out, x_in]).astype(np.float64)
    y = np.hstack([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])

    X += rng.normal(0, noise, X.shape)  # add noise so it's not trivial

    # shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def one_hot(y, num_classes=2):
    """Turn integer labels (n,) into one-hot rows (n, num_classes)."""
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1.0
    return oh


def train_test_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


if __name__ == "__main__":
    X, y = make_moons()
    print("X shape:", X.shape, " y shape:", y.shape)
    print("class balance:", np.bincount(y))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], s=8, label="class 0")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], s=8, label="class 1")
        plt.legend(); plt.title("two moons — not linearly separable")
        plt.savefig("data.png", dpi=110, bbox_inches="tight")
        print("saved data.png")
    except Exception as e:
        print("plot skipped:", e)
