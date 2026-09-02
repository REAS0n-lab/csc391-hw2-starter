"""Row partitioning across ranks.

Kept in its own module, free of any MPI import, so that the partition can be
tested without an MPI runtime present. A sharding bug is silent. Every rank
computes a gradient, the Allreduce succeeds, and the result is the gradient of
a slightly different objective than the one you meant.
"""


def shard_bounds(n, rank, nranks):
    """Half-open row range [start, stop) owned by `rank`.

    The remainder after an even split goes to the lowest ranks, so shard sizes
    differ by at most one. That imbalance is small but it is not nothing at
    high rank counts, and a synchronous iteration finishes when its slowest
    participant does.
    """
    if not 0 <= rank < nranks:
        raise ValueError(f"rank {rank} out of range for {nranks} ranks")
    base, extra = divmod(n, nranks)
    start = rank * base + min(rank, extra)
    stop = start + base + (1 if rank < extra else 0)
    return start, stop


def shard_sizes(n, nranks):
    return [b - a for a, b in (shard_bounds(n, r, nranks) for r in range(nranks))]
