"""Work accounting for the two methods.

The harness calls into this module every time a method performs an update, so
that a convergence curve can be plotted against estimated arithmetic work
rather than only against iterations or wall-clock time. Iterations, epochs,
and work are three different x axes, and a comparison that treats them as
interchangeable is the specific failure this assignment is built around.

The counting functions below are yours to write. Each one returns the work of
a single update in terms of the problem dimensions. Read the implementation of
the method in src/sgd.py or src/rcd.py and count what it actually executes,
not what a textbook says the method costs.
"""

from dataclasses import dataclass


@dataclass
class Work:
    """Accumulated work for a run."""

    flops: float = 0.0
    bytes_moved: float = 0.0
    updates: int = 0

    def add(self, flops, bytes_moved):
        self.flops += flops
        self.bytes_moved += bytes_moved
        self.updates += 1

    def as_dict(self):
        return {"flops": self.flops, "bytes_moved": self.bytes_moved,
                "updates": self.updates}


def sgd_update_cost(n, d, batch_size, dtype_bytes=8):
    """Flops and bytes moved by one mini-batch SGD update.

    Return a (flops, bytes) tuple.

    Count the operations in `sgd.step` in src/sgd.py. There is a matrix-vector
    product against the batch, a residual, a transposed product, the
    regularizer term, and the parameter update. Multiply and add each count as
    one operation.

    For bytes, count the data actually touched. The batch rows are read, the
    parameter vector is read and written. Whether the batch rows are read once
    or twice depends on how the implementation is written, and getting that
    right is part of the exercise.

    TODO. Write this and state your assumptions in the report.
    """
    raise NotImplementedError("sgd_update_cost is yours to derive and write")


def rcd_update_cost(n, d, dtype_bytes=8, maintain_residual=True):
    """Flops and bytes moved by one randomized coordinate descent update.

    Return a (flops, bytes) tuple.

    Count the operations in `rcd.step` in src/rcd.py. With an incrementally
    maintained residual there is one column dot product against the residual,
    one scalar update, and one column-scaled residual update. Without it the
    residual is recomputed from scratch, which is a different cost entirely.
    The `maintain_residual` flag is there so that you can report both and
    justify which one belongs in the comparison.

    TODO. Write this and state your assumptions in the report.
    """
    raise NotImplementedError("rcd_update_cost is yours to derive and write")


def effective_passes(work_flops, n, d, dtype_bytes=8):
    """Convert accumulated flops into equivalent passes over the data.

    One pass over a dense n by d design matrix in a matrix-vector product is
    2*n*d flops. This is the common unit that makes an SGD iteration and a
    coordinate descent iteration comparable, and it is the x axis that the
    plotting script labels "effective data passes".
    """
    return work_flops / (2.0 * n * d)
