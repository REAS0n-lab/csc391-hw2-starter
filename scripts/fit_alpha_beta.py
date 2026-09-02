#!/usr/bin/env python3
"""Fit a linear communication model to ping-pong measurements.

    python3 scripts/fit_alpha_beta.py results/pingpong.csv

The model is

    T(n) = alpha + beta * n

with alpha the latency in seconds and beta the inverse bandwidth in seconds
per byte. The fit is ordinary least squares on the median across trials at
each size.

Two things this script insists on.

It reports the fit over a stated size range and the residual at every size,
including the sizes outside the range. A single alpha and beta pair quoted
without the range it was fitted over is not a usable model, because MPI
implementations switch protocol partway up the size sweep and the two regimes
have different parameters.

It reports where the residual exceeds a threshold, which is the honest answer
to "where does the linear model stop fitting".
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


def load(path):
    per_size = defaultdict(list)
    hosts = set()
    same_node = set()
    with open(path) as fh:
        for row in csv.DictReader(fh):
            per_size[int(row["bytes"])].append(float(row["one_way_seconds"]))
            hosts.add(row.get("rank0_host", ""))
            hosts.add(row.get("rank1_host", ""))
            same_node.add(row.get("same_node", "0"))
    sizes = np.array(sorted(per_size))
    med = np.array([np.median(per_size[s]) for s in sizes])
    lo = np.array([np.min(per_size[s]) for s in sizes])
    hi = np.array([np.max(per_size[s]) for s in sizes])
    counts = np.array([len(per_size[s]) for s in sizes])
    return sizes, med, lo, hi, counts, hosts, same_node


def fit(sizes, times):
    A = np.column_stack([np.ones_like(sizes, dtype=float), sizes.astype(float)])
    coef, *_ = np.linalg.lstsq(A, times, rcond=None)
    return float(coef[0]), float(coef[1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path")
    p.add_argument("--min-bytes", type=int, default=0,
                   help="lower end of the fit range")
    p.add_argument("--max-bytes", type=int, default=2 ** 62,
                   help="upper end of the fit range")
    p.add_argument("--tolerance", type=float, default=0.15,
                   help="relative residual above which the model is called a misfit")
    args = p.parse_args(argv)

    sizes, med, lo, hi, counts, hosts, same_node = load(args.csv_path)
    if len(sizes) < 3:
        print("need at least three message sizes", file=sys.stderr)
        return 1

    if same_node == {"1"}:
        print("WARNING every measurement in this file has both ranks on one "
              "node. These are shared-memory numbers.", file=sys.stderr)
    print(f"hosts seen  {sorted(h for h in hosts if h)}")
    print(f"sizes       {len(sizes)}   trials per size  {counts.min()} to {counts.max()}")
    if counts.min() < 3:
        print("WARNING at least one size has fewer than three trials.", file=sys.stderr)

    mask = (sizes >= args.min_bytes) & (sizes <= args.max_bytes)
    alpha, beta = fit(sizes[mask], med[mask])

    print()
    print(f"fit range   {sizes[mask].min()} to {sizes[mask].max()} bytes "
          f"({mask.sum()} of {len(sizes)} sizes)")
    print(f"alpha       {alpha * 1e6:.3f} us          latency")
    print(f"beta        {beta * 1e9:.6f} ns/byte    inverse bandwidth")
    print(f"1/beta      {1.0 / beta / 1e9:.3f} GB/s        bandwidth")
    print(f"n_half      {alpha / beta:.0f} bytes        size where latency "
          f"and bandwidth terms are equal")

    print()
    print(f"{'bytes':>12} {'measured_us':>13} {'model_us':>11} "
          f"{'rel_resid':>11} {'spread_%':>10} {'in_fit':>7}")
    misfits = []
    for i, s in enumerate(sizes):
        model = alpha + beta * s
        rel = (med[i] - model) / med[i]
        spread = (hi[i] - lo[i]) / med[i] * 100
        flag = "yes" if mask[i] else "no"
        print(f"{s:>12} {med[i]*1e6:>13.3f} {model*1e6:>11.3f} "
              f"{rel:>+11.3f} {spread:>10.1f} {flag:>7}")
        if abs(rel) > args.tolerance:
            misfits.append((s, rel))

    print()
    if misfits:
        print("the linear model misses by more than "
              f"{args.tolerance*100:.0f} percent at these sizes")
        for s, rel in misfits:
            print(f"  {s:>12} bytes   {rel:+.1%}")
        print("A protocol switch, a cache effect, or a change in the number of "
              "network hops will all produce this. Say which one you think it "
              "is and what measurement would distinguish them.")
    else:
        print(f"no size misses by more than {args.tolerance*100:.0f} percent "
              "over the sweep")

    print()
    print("Copy these two lines into prediction.md before running the sweep.")
    print(f"  alpha = {alpha:.6e}   # seconds")
    print(f"  beta  = {beta:.6e}   # seconds per byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
