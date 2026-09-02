#!/usr/bin/env python3
"""Plot predicted against measured strong-scaling behavior.

    python3 scripts/plot_scaling.py results/scaling.jsonl -o figures/scaling.png
    python3 scripts/plot_scaling.py results/scaling.jsonl --alpha 1.6e-6 --beta 8.9e-11

Four panels. Iteration time, speedup, parallel efficiency, and the fraction of
each iteration spent in communication. The last panel is the one that explains
the first three, and it is the reason the harness times the Allreduce
separately.

Passing --alpha and --beta overlays a communication model. The model drawn
here is a ring Allreduce of the gradient vector, which is one common choice
and not necessarily the one your MPI uses. Identifying the actual algorithm is
part of the assignment, and a disagreement between this overlay and your
measurements is evidence rather than an error.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(paths):
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def aggregate(rows, key="iteration_seconds"):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r.get("scale", "unknown")][r["nranks"]].append(r[key])
    return per


def ring_allreduce_seconds(p, nbytes, alpha, beta):
    """Ring Allreduce cost, 2(p-1) messages of nbytes/p each."""
    if p <= 1:
        return 0.0
    return 2 * (p - 1) * (alpha + beta * nbytes / p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="figures/scaling.png")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--dtype-bytes", type=int, default=8)
    args = ap.parse_args(argv)

    rows = load(args.paths)
    if not rows:
        print("no records", file=sys.stderr)
        return 1

    iters = aggregate(rows, "iteration_seconds")
    comms = aggregate(rows, "comm_seconds")
    d = rows[0]["d"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_t, ax_s, ax_e, ax_c = axes.ravel()

    for scale in sorted(iters):
        ps = np.array(sorted(iters[scale]))
        med = np.array([np.median(iters[scale][p]) for p in ps])
        lo = np.array([np.min(iters[scale][p]) for p in ps])
        hi = np.array([np.max(iters[scale][p]) for p in ps])
        runs = np.array([len(iters[scale][p]) for p in ps])
        if runs.min() < 3:
            print(f"warning, scale {scale} has a configuration with "
                  f"{runs.min()} run(s), not 3", file=sys.stderr)

        line, = ax_t.plot(ps, med * 1e3, marker="o", label=f"{scale} measured")
        ax_t.fill_between(ps, lo * 1e3, hi * 1e3, alpha=0.2, color=line.get_color())

        base = med[0]
        speedup = base / med
        ax_s.plot(ps, speedup, marker="o", label=scale, color=line.get_color())
        ax_e.plot(ps, speedup / ps, marker="o", label=scale, color=line.get_color())

        cmed = np.array([np.median(comms[scale][p]) for p in ps])
        ax_c.plot(ps, cmed / med, marker="o", label=scale, color=line.get_color())

        if args.alpha is not None and args.beta is not None:
            nbytes = d * args.dtype_bytes
            compute = med[0] - np.median(comms[scale][ps[0]])
            pred = np.array([
                compute / p + ring_allreduce_seconds(p, nbytes, args.alpha, args.beta)
                for p in ps
            ])
            ax_t.plot(ps, pred * 1e3, linestyle="--", marker="x",
                      color=line.get_color(), label=f"{scale} ring model")
            ax_s.plot(ps, pred[0] / pred, linestyle="--", marker="x",
                      color=line.get_color())
            ax_e.plot(ps, (pred[0] / pred) / ps, linestyle="--", marker="x",
                      color=line.get_color())

    ax_s.plot(sorted({r["nranks"] for r in rows}),
              sorted({r["nranks"] for r in rows}),
              color="0.6", linewidth=0.8, label="ideal")
    ax_e.axhline(1.0, color="0.6", linewidth=0.8)

    for ax, ylabel, title in (
        (ax_t, "iteration time (ms)", "iteration time"),
        (ax_s, "speedup", "speedup against 1 rank"),
        (ax_e, "efficiency", "parallel efficiency"),
        (ax_c, "communication fraction", "share of the iteration in Allreduce"),
    ):
        ax.set_xlabel("ranks")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    ax_t.set_yscale("log")
    ax_c.set_ylim(0, 1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
