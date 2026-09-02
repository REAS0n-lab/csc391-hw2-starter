"""Randomized coordinate descent for ridge regression.

The residual r = X w - y is maintained incrementally. Updating one coordinate
changes the residual by a scaled column, which costs O(n) rather than the
O(n d) a recomputation would cost. That difference is the whole reason the
method is competitive, and it is the first thing a cost model has to capture.
"""

import numpy as np

from . import objective as obj


class RCD:
    """Randomized coordinate descent with exact coordinate minimization.

    Parameters
    ----------
    sampling : {"uniform", "lipschitz"}
        uniform     each coordinate equally likely
        lipschitz   coordinate j drawn with probability proportional to L_j
    step_scale : float
        1.0 gives the exact minimizer along the chosen coordinate. Values
        below 1.0 damp the step, which matters when several coordinates are
        strongly correlated.
    """

    name = "rcd"

    def __init__(self, problem, sampling="uniform", step_scale=1.0, seed=0):
        self.p = problem
        self.rng = np.random.default_rng(seed)
        self.Lj = obj.coordinate_lipschitz(problem)
        self.sampling = sampling
        self.step_scale = float(step_scale)
        if sampling == "lipschitz":
            self.probs = self.Lj / self.Lj.sum()
        else:
            self.probs = None
        self.residual = None
        self.t = 0

    def reset(self, w):
        """Compute the residual once. Every later update adjusts it in place."""
        self.residual = self.p.X @ w - self.p.y

    def step(self, w):
        """Apply one coordinate update in place and return the chosen index."""
        if self.residual is None:
            self.reset(w)
        j = int(self.rng.choice(self.p.d, p=self.probs)) if self.probs is not None \
            else int(self.rng.integers(0, self.p.d))

        xj = self.p.X[:, j]
        grad_j = float(np.dot(xj.astype(np.float64),
                              self.residual.astype(np.float64))) / self.p.n \
            + self.p.lam * float(w[j])
        delta = -self.step_scale * grad_j / self.Lj[j]
        w[j] += delta
        self.residual += delta * xj
        self.t += 1
        return j

    def updates_per_epoch(self):
        """d coordinate updates together touch every entry of X once."""
        return self.p.d
