"""Least-squares problem instances.

Two families are provided. A synthetic family with a controllable condition
number and feature scaling, and a loader for LIBSVM sparse text files.

The synthetic family exists so that the true minimizer is known in closed
form, which lets every convergence plot use a real objective gap rather than
a gap against the best value the run happened to reach.
"""

import gzip
import io
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Problem:
    """A ridge regression instance,  min_w  0.5/n ||Xw - y||^2 + 0.5 lam ||w||^2."""

    X: np.ndarray
    y: np.ndarray
    lam: float
    name: str
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def d(self):
        return self.X.shape[1]

    def describe(self):
        return (f"{self.name}  n={self.n}  d={self.d}  lam={self.lam:g}  "
                f"dtype={self.X.dtype}")


def standardize(X, eps=1e-12):
    """Center each column and scale it to unit standard deviation.

    Column scaling changes the conditioning of the problem and therefore
    changes how both methods behave. It is applied by default because a
    comparison run on unscaled features measures the scaling as much as the
    algorithms. Turn it off with standardize=False if that is the effect you
    want to study, and say so in the report.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def synthetic(n=4000, d=200, condition=100.0, noise=0.1, seed=0,
              lam=1e-3, scale_features=True, dtype=np.float64):
    """Generate a dense least-squares problem with a prescribed condition number.

    The design matrix is U S V^T with singular values spaced logarithmically
    between 1 and 1/condition, so the Gram matrix has condition number
    condition^2 before regularization.
    """
    rng = np.random.default_rng(seed)
    k = min(n, d)
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((d, k)))
    s = np.logspace(0.0, -np.log10(condition), k)
    X = (U * s) @ V.T
    X = np.ascontiguousarray(X.astype(dtype))

    w_true = rng.standard_normal(d).astype(dtype)
    y = X @ w_true + noise * rng.standard_normal(n).astype(dtype)

    meta = {"condition_requested": condition, "noise": noise, "seed": seed,
            "w_true_norm": float(np.linalg.norm(w_true))}
    if scale_features:
        X, mu, sigma = standardize(X)
        meta["standardized"] = True
    y = y - y.mean()

    return Problem(X=X, y=y.astype(dtype), lam=lam,
                   name=f"synthetic-n{n}-d{d}-k{condition:g}", meta=meta)


def load_libsvm(path, d=None, dtype=np.float64, scale_features=True, lam=1e-3):
    """Read a LIBSVM format text file into a dense Problem.

    The format is one example per line,  label idx:value idx:value ...  with
    one-based indices. Dense storage is used because the problem sizes in this
    course are small enough for it and because a dense array keeps the cost
    model in HW1 simple. That choice stops being defensible on a real sparse
    dataset, which is worth one sentence in the limitations section.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    labels, rows = [], []
    max_idx = 0
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            labels.append(float(parts[0]))
            entries = []
            for token in parts[1:]:
                idx, _, val = token.partition(":")
                if not val:
                    continue
                i = int(idx) - 1
                entries.append((i, float(val)))
                max_idx = max(max_idx, i)
            rows.append(entries)

    d = d or (max_idx + 1)
    X = np.zeros((len(rows), d), dtype=dtype)
    for r, entries in enumerate(rows):
        for i, v in entries:
            if i < d:
                X[r, i] = v
    y = np.asarray(labels, dtype=dtype)

    meta = {"path": str(path), "nnz_fraction": float((X != 0).mean())}
    if scale_features:
        X, _, _ = standardize(X)
        meta["standardized"] = True
    y = y - y.mean()
    return Problem(X=X, y=y, lam=lam, name=f"libsvm-{path}", meta=meta)


SCALES = {
    "small":  dict(n=2000,  d=100,  condition=50.0),
    "medium": dict(n=8000,  d=400,  condition=100.0),
    "large":  dict(n=32000, d=800,  condition=200.0),
}


def by_scale(scale, seed=0, lam=1e-3):
    """Return one of the three named problem scales.

    CSC 391 uses two of the three. CSC 691 uses all three and reports whether
    the conclusion survives.
    """
    if scale not in SCALES:
        raise KeyError(f"unknown scale {scale!r}, choose from {sorted(SCALES)}")
    return synthetic(seed=seed, lam=lam, **SCALES[scale])
