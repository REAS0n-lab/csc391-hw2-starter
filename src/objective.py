"""Ridge regression objective, gradients, and the closed-form minimizer.

The objective is

    f(w) = (1 / 2n) * ||X w - y||^2  +  (lam / 2) * ||w||^2

Every accumulation happens in float64 even when the data is float32. A sum of
n residuals in float32 loses accuracy at exactly the point where the objective
gap gets small, which is the region every convergence plot in this assignment
cares about. Storage precision and accumulation precision are separate
choices, and this file makes the second one explicitly.
"""

import numpy as np


def objective(problem, w):
    """Return f(w) as a Python float."""
    r = problem.X @ w - problem.y
    quad = float(np.dot(r.astype(np.float64), r.astype(np.float64)))
    reg = float(np.dot(w.astype(np.float64), w.astype(np.float64)))
    return 0.5 * quad / problem.n + 0.5 * problem.lam * reg


def gradient(problem, w):
    """Full gradient of f at w."""
    r = problem.X @ w - problem.y
    return (problem.X.T @ r) / problem.n + problem.lam * w


def stochastic_gradient(problem, w, idx):
    """Gradient of the objective restricted to the examples in idx.

    The regularizer is included in full, which is the standard convention for
    an averaged mini-batch gradient. If you change that convention, say so,
    because it changes the effective step size.
    """
    Xb = problem.X[idx]
    r = Xb @ w - problem.y[idx]
    return (Xb.T @ r) / len(idx) + problem.lam * w


def coordinate_derivative(problem, w, j, residual=None):
    """Partial derivative of f with respect to w[j].

    Pass the current residual X w - y to avoid recomputing it. Maintaining the
    residual incrementally is what makes coordinate descent cheap, and the
    difference between the two paths is one of the things the cost model in
    this assignment has to account for.
    """
    r = residual if residual is not None else (problem.X @ w - problem.y)
    xj = problem.X[:, j]
    return float(np.dot(xj.astype(np.float64), r.astype(np.float64))) / problem.n \
        + problem.lam * float(w[j])


def coordinate_lipschitz(problem):
    """Per-coordinate curvature,  L_j = ||x_j||^2 / n + lam.

    This is the exact second derivative of f in coordinate j, so an exact
    coordinate minimization step divides by it.
    """
    col_sq = np.einsum("ij,ij->j", problem.X.astype(np.float64),
                       problem.X.astype(np.float64))
    return col_sq / problem.n + problem.lam


def global_lipschitz(problem):
    """Largest eigenvalue of the Hessian,  ||X||_2^2 / n + lam."""
    s = np.linalg.svd(problem.X.astype(np.float64), compute_uv=False)
    return float(s[0] ** 2) / problem.n + problem.lam


def strong_convexity(problem):
    """Smallest eigenvalue of the Hessian,  sigma_min(X)^2 / n + lam."""
    s = np.linalg.svd(problem.X.astype(np.float64), compute_uv=False)
    smallest = s[-1] if problem.d <= problem.n else 0.0
    return float(smallest ** 2) / problem.n + problem.lam


def solve_exact(problem):
    """Closed-form minimizer,  (X^T X / n + lam I)^{-1} X^T y / n.

    Used for the objective gap f(w) - f(w*). Solving the normal equations
    squares the condition number, which is acceptable here because the
    regularizer bounds it and because this value is a reference rather than a
    method under study. On a badly conditioned problem, prefer an SVD or QR
    based solve and note the change.
    """
    X = problem.X.astype(np.float64)
    y = problem.y.astype(np.float64)
    A = (X.T @ X) / problem.n + problem.lam * np.eye(problem.d)
    b = (X.T @ y) / problem.n
    return np.linalg.solve(A, b)
