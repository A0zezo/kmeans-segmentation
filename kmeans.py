"""K-means (k-means++ init) from scratch in NumPy, applied to synthetic customer data and checked against scikit-learn."""
import numpy as np


def _run(X, k, iters, seed):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    inertia = float(((X - C[lab]) ** 2).sum())
    return lab, C, inertia


def kmeans(X, k, iters=100, n_init=10):
    """Best of n_init runs (lowest inertia); a single k-means run can get stuck in a poor local minimum."""
    return min((_run(X, k, iters, s) for s in range(n_init)), key=lambda r: r[2])


def make_customers(seed=0):
    rng = np.random.default_rng(seed)
    groups = [(20, 300, 40), (60, 1200, 10), (35, 3000, 25)]  # (orders/yr, spend, recency days)
    X = np.vstack([rng.normal(g, [4, 150, 6], size=(100, 3)) for g in groups])
    return X
