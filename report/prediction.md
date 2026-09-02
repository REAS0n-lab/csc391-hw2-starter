# HW2 prediction record

Fill this in **before** submitting any multi-node training job. A model built
after the scaling curve can be adjusted until it reproduces the curve, and a
model that can reproduce any curve has predicted nothing. This file is the
separation between the two sources of evidence, and it is graded on being
written first rather than on being right.

Commit this file, then run the sweep. The commit timestamp is the record.

---

## 1. Communication parameters

From `scripts/fit_alpha_beta.py` on the ping-pong measurements. Independent of
any training run.

| Field | Value |
|---|---|
| alpha, latency (s) | |
| beta, inverse bandwidth (s/byte) | |
| Fit range (bytes) | |
| Sizes where the linear model misses by more than 15 percent | |
| Ranks on distinct nodes | yes / no |
| Trials per size | |
| Ping-pong job id | |

CSC 691 fills this in twice, once for two ranks on one node and once for two
ranks on two nodes, and states which pair belongs in the multi-node model and
why.

| Field | Intra-node | Inter-node |
|---|---|---|
| alpha (s) | | |
| beta (s/byte) | | |

## 2. Iteration-time model

Write the model before filling in the table below.

```
T(p) = T_compute(p) + T_comm(p)

T_compute(p) = ...

T_comm(p)    = ...
```

Answer each of these in the model.

- How much arithmetic does one rank do per iteration, in terms of the local
  batch size, d, and p.
- How many bytes does the Allreduce move, and how does that depend on p.
- Which Allreduce algorithm are you assuming. Name it, and cite where you
  found that your MPI uses it. `ompi_info --param coll all --level 9`, the
  `MPI_ALLREDUCE` section of your implementation's documentation, or an
  `MPICH_*` environment setting are all acceptable sources. A guess is not,
  and the guess and the citation give different answers for the p dependence.
- What does the model leave out.

| Term | Expression | Value at p = 1 |
|---|---|---|
| Flops per iteration per rank | | |
| Bytes reduced per iteration | | |
| Allreduce algorithm assumed | | |
| Source for that assumption | | |
| Omitted from the model | | |

## 3. Predictions

Fill every cell before the sweep runs. Leave nothing blank.

### Scale A, name it here

| Nodes | Predicted iteration time (ms) | Predicted speedup | Predicted efficiency |
|---:|---:|---:|---:|
| 1 | | 1.00 | 1.00 |
| 2 | | | |
| 4 | | | |
| 8 | | | |

### Scale B, name it here

| Nodes | Predicted iteration time (ms) | Predicted speedup | Predicted efficiency |
|---:|---:|---:|---:|
| 1 | | 1.00 | 1.00 |
| 2 | | | |
| 4 | | | |
| 8 | | | |

### Useful scaling limit

The node count beyond which the model says added nodes stop paying for
themselves, and the criterion you used to define that.

| Scale | Limit (nodes) | Criterion |
|---|---:|---|
| A | | |
| B | | |

## 4. Rules fixed before the sweep

These decide what the speedup number means. Fixing them afterwards is how a
scaling study becomes unfalsifiable.

| Rule | Choice |
|---|---|
| Strong or weak scaling | |
| Global batch held fixed, or local batch held fixed | |
| Iterations per measurement | |
| Warmup iterations discarded | |
| Repetitions per configuration | at least 3 |
| Which rank's time is reported | |
| Target quality, if any, and how it is measured | |

The starter reports the maximum across ranks, because a synchronous iteration
finishes when its slowest participant does. If you report something else, say
so here and justify it.

## 5. Sign-off

| Field | Value |
|---|---|
| Prediction completed at | |
| Git commit of this file | |
| First training job submitted at | |
