"""Synchronous data-parallel mini-batch SGD across MPI ranks.

Each rank owns a contiguous shard of the rows of X. One iteration is

  1. every rank draws a local mini-batch from its own shard and computes the
     gradient of the objective restricted to that batch,
  2. an Allreduce sums the local gradients across ranks,
  3. every rank divides by the number of ranks and applies the same update.

Every rank therefore holds the same parameter vector at the end of every
iteration, which makes this mathematically equivalent to serial mini-batch SGD
with a global batch of `local_batch * nranks`. That equivalence is what makes
a strong-scaling comparison meaningful, and it is worth confirming rather than
assuming. `--check-equivalence` does exactly that.

Compute time and communication time are measured separately. The barrier
before the Allreduce timer is deliberate. Without it, the Allreduce absorbs
whatever load imbalance preceded it and the communication measurement becomes
a measurement of the slowest rank's compute.
"""

import argparse
import json
import os
import socket
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import objective as obj, problems  # noqa: E402
from src.sharding import shard_bounds  # noqa: E402


class DistributedSGD:
    def __init__(self, problem, comm, local_batch=32, step0=None,
                 schedule="constant", seed=0):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nranks = comm.Get_size()
        self.global_problem = problem

        start, stop = shard_bounds(problem.n, self.rank, self.nranks)
        self.Xl = np.ascontiguousarray(problem.X[start:stop])
        self.yl = np.ascontiguousarray(problem.y[start:stop])
        self.local_n = stop - start
        self.lam = problem.lam
        self.d = problem.d

        self.local_batch = min(int(local_batch), self.local_n)
        self.global_batch = self.local_batch * self.nranks
        self.rng = np.random.default_rng(seed + 7919 * self.rank)

        L = obj.global_lipschitz(problem)
        self.step0 = float(step0) if step0 is not None else 1.0 / L
        self.schedule = schedule
        self.mu = obj.strong_convexity(problem)
        self.t = 0

        self.grad_buf = np.zeros(self.d, dtype=np.float64)
        self.recv_buf = np.zeros(self.d, dtype=np.float64)

    def step_size(self):
        if self.schedule == "constant":
            return self.step0
        return self.step0 / (1.0 + self.step0 * self.mu * self.t)

    def local_gradient(self, w):
        idx = self.rng.integers(0, self.local_n, size=self.local_batch)
        Xb = self.Xl[idx]
        r = Xb @ w - self.yl[idx]
        return (Xb.T @ r) / self.local_batch + self.lam * w

    def step(self, w):
        """One iteration. Returns (compute_seconds, comm_seconds)."""
        t0 = MPI.Wtime()
        self.grad_buf[:] = self.local_gradient(w)
        t1 = MPI.Wtime()

        # The barrier keeps load imbalance out of the communication timer.
        self.comm.Barrier()
        t2 = MPI.Wtime()
        self.comm.Allreduce(self.grad_buf, self.recv_buf, op=MPI.SUM)
        t3 = MPI.Wtime()

        w -= self.step_size() * (self.recv_buf / self.nranks)
        self.t += 1
        return (t1 - t0), (t3 - t2)


def check_equivalence(problem, comm, local_batch, seed):
    """Confirm the averaged Allreduce gradient equals a global batch gradient.

    Each rank builds its local batch, the ranks gather the indices, and rank 0
    compares the reduced gradient with the gradient computed serially on the
    union of those indices. A mismatch here invalidates every scaling number
    that follows it.
    """
    m = DistributedSGD(problem, comm, local_batch=local_batch, seed=seed)
    w = np.full(problem.d, 0.1)

    idx = m.rng.integers(0, m.local_n, size=m.local_batch)
    start, _ = shard_bounds(problem.n, m.rank, m.nranks)
    global_idx = idx + start

    Xb = m.Xl[idx]
    r = Xb @ w - m.yl[idx]
    local = (Xb.T @ r) / m.local_batch + m.lam * w

    reduced = np.zeros_like(local)
    comm.Allreduce(local, reduced, op=MPI.SUM)
    reduced /= m.nranks

    all_idx = comm.gather(global_idx, root=0)
    ok = True
    if m.rank == 0:
        union = np.concatenate(all_idx)
        serial = obj.stochastic_gradient(problem, w, union)
        err = float(np.linalg.norm(reduced - serial) / max(np.linalg.norm(serial), 1e-30))
        ok = err < 1e-10
        print(f"[equivalence] relative error {err:.3e}  ->  {'ok' if ok else 'MISMATCH'}")
    return comm.bcast(ok, root=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", default="medium", choices=sorted(problems.SCALES))
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--local-batch", type=int, default=32)
    p.add_argument("--global-batch", type=int, default=None,
                   help="hold the global batch fixed, splitting it across ranks")
    p.add_argument("--step0", type=float, default=None)
    p.add_argument("--schedule", default="constant", choices=["constant", "inverse"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--check-equivalence", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    problem = problems.by_scale(args.scale, seed=args.seed)

    local_batch = args.local_batch
    if args.global_batch is not None:
        local_batch = max(1, args.global_batch // nranks)

    hosts = comm.gather(socket.gethostname(), root=0)
    if rank == 0:
        unique = sorted(set(hosts))
        print(f"ranks {nranks}   distinct hosts {len(unique)}   {unique}")
        print(f"{problem.describe()}   local_batch {local_batch}   "
              f"global_batch {local_batch * nranks}")
        if len(unique) < min(nranks, 2):
            print("WARNING every rank is on one host. This is not a "
                  "multi-node measurement.", file=sys.stderr)

    if args.check_equivalence:
        if not check_equivalence(problem, comm, local_batch, args.seed):
            comm.Abort(5)

    records = []
    for rep in range(args.repeat):
        m = DistributedSGD(problem, comm, local_batch=local_batch,
                           step0=args.step0, schedule=args.schedule,
                           seed=args.seed + 100 * rep)
        w = np.zeros(problem.d, dtype=np.float64)

        for _ in range(args.warmup):
            m.step(w)

        comp = np.zeros(args.iters)
        comm_t = np.zeros(args.iters)
        comm.Barrier()
        wall0 = MPI.Wtime()
        for i in range(args.iters):
            comp[i], comm_t[i] = m.step(w)
        wall = MPI.Wtime() - wall0

        # Report the slowest rank. A mean across ranks hides imbalance, and
        # a synchronous iteration finishes when its slowest participant does.
        stats = np.array([comp.sum(), comm_t.sum(), wall])
        worst = np.zeros(3)
        comm.Allreduce(stats, worst, op=MPI.MAX)

        f = obj.objective(problem, w)
        if rank == 0:
            rec = {
                "nranks": nranks,
                "hosts": len(set(hosts)),
                "scale": args.scale,
                "n": problem.n,
                "d": problem.d,
                "local_batch": local_batch,
                "global_batch": local_batch * nranks,
                "iters": args.iters,
                "repeat": rep,
                "iteration_seconds": worst[2] / args.iters,
                "compute_seconds": worst[0] / args.iters,
                "comm_seconds": worst[1] / args.iters,
                "wall_seconds": worst[2],
                "objective": f,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            }
            records.append(rec)
            print(f"  rep {rep}  iter {rec['iteration_seconds']*1e3:8.4f} ms  "
                  f"compute {rec['compute_seconds']*1e3:8.4f} ms  "
                  f"comm {rec['comm_seconds']*1e3:8.4f} ms  f {f:.6e}")

    if rank == 0 and args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(records)} records to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
