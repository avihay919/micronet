# MicroNet — a neural network from scratch (NumPy)

A small multi-layer perceptron implemented from first principles — forward pass
and backpropagation written by hand, no deep-learning frameworks. Trained on a
nonlinear "two moons" classification task, with an experiment comparing
different architectures.

## Why this project
Implementing backprop by hand forces you to understand what a network actually
does, rather than calling `model.fit()`. The experiment section explores how
architecture choices (depth, width, activation, skip connections) change what
the network can learn.

## Run
```bash
python3 data.py        # generate + visualise the dataset (data.png)
python3 micronet.py    # train the network
```

## Files
- `data.py` — synthetic two-moons dataset (done)
- `micronet.py` — the MLP: forward + backprop (the core you implement)
- `experiments.py` — architecture comparison + plots (Day 2)

---

## 2-day plan

**Day 1 — get a working network (the hard, important part)**
1. Run `data.py`, look at `data.png`, understand why a linear model can't solve it.
2. Implement `relu`, then `forward` in `micronet.py`. Sanity-check shapes.
3. Implement `backward` using the formulas in the comments. This is the meat.
4. Run `micronet.py`. Tune `lr`/`epochs` until you clear ~95% test accuracy.
   - If it won't learn: print gradient shapes, check `dz_last = (probs - Y)/N`,
     verify the ReLU derivative is `(z > 0)`.

**Day 2 — experiments + write-up (the "creativity" signal)**
5. Compare architectures and record results in a small table:
   - depth: `[2,2]` (no hidden) vs `[2,16,2]` vs `[2,16,16,2]` vs `[2,32,32,32,2]`
   - width: 4 vs 16 vs 64 hidden units
   - activation: ReLU vs tanh (try swapping it in — note the derivative changes)
   - **a skip/residual connection**: add the input (or an earlier activation) back
     in before the final layer and see if deeper nets train more easily.
6. Plot the decision boundary for the best model (grid of points -> predict ->
   contour) and the loss curve. Write 4–5 sentences on what you found and why.

**Optional stretch (only if Day 2 is fast) — the OS/optimization angle**
7. `matmul.c`: implement matrix multiply twice — naive triple loop vs a
   cache-blocked (tiled) version — and time both on a large matrix.
   Matmul is the core operation inside `forward`, so this connects the AI
   architecture to systems-level performance. Report the speedup.
   This is the bit that makes you stand out to a team doing *both* OS and AI.

## What you should be able to explain afterwards
- What backprop computes and why the chain rule gives those formulas
- Why softmax+cross-entropy collapses to `dz = probs - y`
- What the hidden layers buy you over a linear model
- Which architecture worked best and your hypothesis for why

That list is basically your interview prep. If you can talk through it, you can
defend this project cold.
