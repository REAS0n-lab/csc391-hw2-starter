# CSC 391/691 HW2 starter

A ping-pong benchmark in C and in mpi4py, a synchronous data-parallel SGD
implementation with the Allreduce timed separately from the compute, a fit
script that turns ping-pong measurements into a latency and bandwidth pair,
and the prediction record you fill in before the sweep runs.

The training code works. The assignment is to predict what it will do on 1, 2,
4, and 8 nodes, then explain where the prediction was wrong.

**Opens Wednesday 9/23. Due Wednesday 10/7 at 2:00 p.m.**

## Layout

```
pingpong/pingpong.c        two-process latency and bandwidth benchmark, C
pingpong/pingpong.py       the same measurement in mpi4py
pingpong/Makefile          make, or make check to compile without linking
src/dist_sgd.py            synchronous data-parallel SGD, compute and comm timed apart
src/                       problem, objective, and serial methods carried from HW1
scripts/fit_alpha_beta.py  fits T(n) = alpha + beta n and reports where it stops fitting
scripts/plot_scaling.py    iteration time, speedup, efficiency, communication fraction
report/prediction.md       fill this in and commit it before the sweep
jobs/pingpong.slurm        two distinct nodes
jobs/pingpong_intra.slurm  both ranks on one node, the CSC 691 condition
jobs/scaling_sweep.slurm   array over 1, 2, 4, 8 nodes at two problem scales
```

## Build and quick check

```bash
make -C pingpong
python3 -m pytest tests/test_offline.py -q
```

`make check` compiles without linking, which is enough to confirm a toolchain
when no allocation is available. `MPICC=mpiicc make` overrides the wrapper.

## Order of operations

The order matters and it is graded.

**1. Measure the machine.** Run the ping-pong on two distinct nodes.

```bash
./submit.sh cpu jobs/pingpong.slurm
```

The benchmark prints both hostnames and warns when both ranks landed on one
node. A run with both ranks on one node measures shared memory, which is a
different machine, and its parameters do not describe the fabric.

**2. Fit the parameters.**

```bash
python3 scripts/fit_alpha_beta.py results/pingpong-inter.csv
```

The output holds alpha in seconds, beta in seconds per byte, the size range
the fit covers, and the sizes where the linear model misses by more than 15
percent. Report the range. MPI implementations switch protocol partway up a
size sweep and the two regimes have different parameters, so a single pair
quoted without its range is not a usable model.

**3. Predict.** Fill in every cell of `report/prediction.md` and commit it. A
model built after the scaling curve can be adjusted until it reproduces the
curve, and a model that can reproduce any curve has predicted nothing.

**4. Sweep.**

```bash
./submit.sh cpu jobs/scaling_sweep.slurm
```

**5. Explain the disagreement.**

```bash
python3 scripts/plot_scaling.py results/scaling.jsonl -o figures/scaling.png \
        --alpha <your alpha> --beta <your beta>
```

## What the training code measures, and why that way

**Compute and communication are timed separately, with a barrier between
them.** Without the barrier the Allreduce absorbs whatever load imbalance
preceded it, and the communication measurement becomes a measurement of the
slowest rank's compute.

**The reported time is the maximum across ranks, not the mean.** A synchronous
iteration finishes when its slowest participant does. A mean across ranks
reports a number no rank experienced.

**`--check-equivalence` verifies the reduced gradient.** Each rank builds a
local batch, the union of the batch indices is gathered, and the averaged
Allreduce result is compared against the gradient computed serially on that
union. A mismatch invalidates every scaling number that follows it, and the
sweep job runs the check on every task.

**`--global-batch` holds the total batch fixed across node counts.** That is
what makes the comparison strong scaling. Holding the local batch fixed
instead grows the problem with the node count, which is weak scaling and a
different claim. Choose one, record it in `prediction.md`, and do not change
it after seeing the curve.

**One BLAS thread per rank.** The sweep job sets `OMP_NUM_THREADS=1`.
Otherwise a one-rank run quietly uses every core on the node and the eight-node
speedup is measured against a baseline that was already parallel.

## Identifying the collective

The model in `plot_scaling.py` assumes a ring Allreduce, which costs
`2(p-1)(alpha + beta * bytes / p)`. That is one common implementation and not
necessarily the one your MPI uses. Recursive doubling has a different
dependence on p, and the two predictions diverge as p grows.

Find out which one you have and cite the source.

```bash
ompi_info --param coll all --level 9 | grep -i allreduce   # Open MPI
mpichversion                                                # MPICH
```

A disagreement between the overlay and your measurements is evidence to
explain, not an error to fix.

## Node counts

`jobs/scaling_sweep.slurm` sweeps 1, 2, 4, and 8. If the course allocation
caps lower, set `DEAC_MAX_NODES` in `deac/site.sh`, shorten the `NODES` array
to match, and say in the report which counts were available. A curve that
stops at 4 nodes with the reason stated is worth more than a curve to 8 that
waited three days in a queue.

## CSC 691 condition

Measure both intra-node and inter-node communication parameters
(`jobs/pingpong_intra.slurm` gives the first), then justify which pair belongs
in the multi-node model. The answer is not automatic. An 8-rank job on 8 nodes
uses the fabric, but the same 8 ranks packed onto one node do not, and a model
built from the wrong pair will be wrong in a specific and explainable
direction.
