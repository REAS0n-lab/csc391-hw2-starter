"""Mini-batch stochastic gradient descent for ridge regression."""

import numpy as np

from . import objective as obj


class SGD:
    """Mini-batch SGD with a configurable step-size schedule.

    Parameters
    ----------
    step0 : float or None
        Initial step size. None selects 1 / L where L is the global Lipschitz
        constant, which is a defensible default rather than a tuned value.
    schedule : {"constant", "inverse", "sqrt"}
        constant  eta_t = step0
        inverse   eta_t = step0 / (1 + step0 * mu * t)
        sqrt      eta_t = step0 / sqrt(1 + t)
    batch_size : int
        Examples per update.
    with_replacement : bool
        True draws each batch independently. False shuffles once per epoch and
        walks through, which is what most implementations do and what most
        theory does not cover. The difference is small and worth a sentence.
    """

    name = "sgd"

    def __init__(self, problem, step0=None, schedule="inverse", batch_size=1,
                 with_replacement=False, seed=0):
        self.p = problem
        self.rng = np.random.default_rng(seed)
        self.batch_size = int(batch_size)
        self.schedule = schedule
        self.with_replacement = with_replacement
        self.L = obj.global_lipschitz(problem)
        self.mu = obj.strong_convexity(problem)
        self.step0 = float(step0) if step0 is not None else 1.0 / self.L
        self.t = 0
        self._perm = None
        self._cursor = 0

    def step_size(self):
        if self.schedule == "constant":
            return self.step0
        if self.schedule == "sqrt":
            return self.step0 / np.sqrt(1.0 + self.t)
        return self.step0 / (1.0 + self.step0 * self.mu * self.t)

    def _next_batch(self):
        n = self.p.n
        if self.with_replacement:
            return self.rng.integers(0, n, size=self.batch_size)
        if self._perm is None or self._cursor + self.batch_size > n:
            self._perm = self.rng.permutation(n)
            self._cursor = 0
        idx = self._perm[self._cursor:self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return idx

    def step(self, w):
        """Apply one update in place and return the step size used."""
        idx = self._next_batch()
        g = obj.stochastic_gradient(self.p, w, idx)
        eta = self.step_size()
        w -= eta * g
        self.t += 1
        return eta

    def updates_per_epoch(self):
        """Updates that together touch n examples."""
        return max(1, self.p.n // self.batch_size)
