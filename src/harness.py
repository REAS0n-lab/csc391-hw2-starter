"""Run a method on a problem and record a convergence history.

One record is written per checkpoint, holding the objective gap alongside four
different measures of progress. Iterations, effective data passes, estimated
arithmetic work, and wall-clock time are four separate x axes, and a method
can win on one and lose on another. Recording all four in the same run is what
makes that visible.

Timing excludes the checkpoint itself. Evaluating the full objective costs a
matrix-vector product, and charging that to the method under study would make
a frequently checkpointed run look slower than it is.
"""

import json
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import costs, objective as obj


@dataclass
class Checkpoint:
    iteration: int
    epoch: float
    objective: float
    gap: float
    grad_norm: float
    seconds: float
    flops: float
    effective_passes: float


@dataclass
class RunResult:
    method: str
    problem: str
    config: dict
    checkpoints: list = field(default_factory=list)
    environment: dict = field(default_factory=dict)
    stopped_by: str = ""
    w_final: object = None

    def to_records(self):
        base = {"method": self.method, "problem": self.problem,
                "stopped_by": self.stopped_by}
        base.update({f"config.{k}": v for k, v in self.config.items()})
        base.update({f"env.{k}": v for k, v in self.environment.items()})
        out = []
        for c in self.checkpoints:
            row = dict(base)
            row.update(asdict(c))
            out.append(row)
        return out


def environment():
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }


def gap_tolerance_reached(gap, tol):
    return gap <= tol


def grad_tolerance_reached(grad_norm, tol):
    return grad_norm <= tol


def run(method, problem, max_epochs=50, checkpoints_per_epoch=4,
        f_star=None, gap_tol=None, grad_tol=None, time_budget=None,
        count_work=True, seed=0):
    """Run `method` on `problem` and return a RunResult.

    Stopping is whichever of the following fires first. Which rule you apply,
    and whether it treats the two methods fairly, is part of the assignment.

      max_epochs     a fixed budget of effective data passes
      gap_tol        objective gap f(w) - f(w*) below a threshold
      grad_tol       full gradient norm below a threshold
      time_budget    wall-clock seconds
    """
    d = problem.d
    w = np.zeros(d, dtype=problem.X.dtype)
    if hasattr(method, "reset"):
        method.reset(w)

    if f_star is None:
        f_star = obj.objective(problem, obj.solve_exact(problem))

    per_epoch = method.updates_per_epoch()
    interval = max(1, per_epoch // max(1, checkpoints_per_epoch))
    total_updates = int(max_epochs * per_epoch)

    work = costs.Work()
    elapsed = 0.0
    stopped_by = "max_epochs"
    result = RunResult(method=method.name, problem=problem.name,
                       config=_config(method, problem, max_epochs, gap_tol,
                                      grad_tol, time_budget, seed),
                       environment=environment())

    def checkpoint(it):
        f = obj.objective(problem, w)
        g = float(np.linalg.norm(obj.gradient(problem, w)))
        result.checkpoints.append(Checkpoint(
            iteration=it,
            epoch=it / per_epoch,
            objective=f,
            gap=max(f - f_star, 0.0),
            grad_norm=g,
            seconds=elapsed,
            flops=work.flops,
            effective_passes=costs.effective_passes(work.flops, problem.n, problem.d)
            if work.flops else it / per_epoch,
        ))
        return f - f_star, g

    checkpoint(0)

    for it in range(1, total_updates + 1):
        start = time.perf_counter()
        method.step(w)
        elapsed += time.perf_counter() - start

        if count_work:
            try:
                if method.name == "sgd":
                    f_c, b_c = costs.sgd_update_cost(problem.n, problem.d,
                                                     method.batch_size)
                else:
                    f_c, b_c = costs.rcd_update_cost(problem.n, problem.d)
                work.add(f_c, b_c)
            except NotImplementedError:
                count_work = False

        if it % interval == 0 or it == total_updates:
            gap, grad_norm = checkpoint(it)
            if gap_tol is not None and gap_tolerance_reached(gap, gap_tol):
                stopped_by = "gap_tol"
                break
            if grad_tol is not None and grad_tolerance_reached(grad_norm, grad_tol):
                stopped_by = "grad_tol"
                break
            if time_budget is not None and elapsed >= time_budget:
                stopped_by = "time_budget"
                break

    result.stopped_by = stopped_by
    result.w_final = w
    return result


def _config(method, problem, max_epochs, gap_tol, grad_tol, time_budget, seed):
    cfg = {"max_epochs": max_epochs, "gap_tol": gap_tol, "grad_tol": grad_tol,
           "time_budget": time_budget, "seed": seed,
           "n": problem.n, "d": problem.d, "lam": problem.lam}
    for key in ("batch_size", "schedule", "step0", "with_replacement",
                "sampling", "step_scale"):
        if hasattr(method, key):
            cfg[key] = getattr(method, key)
    return cfg


def write_jsonl(results, path):
    """Append every checkpoint of every result to a JSON Lines file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        for r in results:
            for row in r.to_records():
                fh.write(json.dumps(row) + "\n")
    return path
