"""Offline tests. No MPI runtime and no cluster required."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


from src.sharding import shard_bounds, shard_sizes  # noqa: E402


@pytest.mark.parametrize("n,p", [(100, 1), (100, 3), (100, 7), (8000, 8), (5, 8)])
def test_shards_partition_the_rows(n, p):
    bounds = [shard_bounds(n, r, p) for r in range(p)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    for (_, stop), (start, _) in zip(bounds, bounds[1:]):
        assert stop == start
    sizes = [b - a for a, b in bounds]
    assert sum(sizes) == n
    assert max(sizes) - min(sizes) <= 1


def test_dist_sgd_uses_the_same_partition():
    """dist_sgd must partition rows with the tested helper, not its own copy.
    Skipped where mpi4py is absent."""
    pytest.importorskip("mpi4py")
    from src.dist_sgd import shard_bounds as used
    assert used is shard_bounds


def test_shard_sizes_differ_by_at_most_one():
    for n in (7, 100, 8000):
        for p in (1, 3, 8, 16):
            sizes = shard_sizes(n, p)
            assert sum(sizes) == n
            assert max(sizes) - min(sizes) <= 1


def test_out_of_range_rank_is_rejected():
    with pytest.raises(ValueError):
        shard_bounds(100, 4, 4)


def test_pingpong_c_compiles():
    """mpicc must build the benchmark. Skipped where no MPI compiler exists."""
    import shutil
    mpicc = os.environ.get("MPICC", "mpicc")
    if shutil.which(mpicc) is None:
        pytest.skip(f"{mpicc} not on PATH")
    r = subprocess.run(["make", "-C", os.path.join(ROOT, "pingpong"), "check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_fit_recovers_known_parameters(tmp_path):
    import csv
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fit", os.path.join(ROOT, "scripts", "fit_alpha_beta.py"))
    fit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_mod)

    alpha, beta = 2e-6, 1e-10
    path = tmp_path / "pp.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bytes", "trial", "rep", "one_way_seconds",
                    "rank0_host", "rank1_host", "same_node"])
        for n in (8, 64, 512, 4096, 32768, 262144, 1048576):
            for t in range(3):
                w.writerow([n, t, 10, f"{alpha + beta * n:.12e}", "a", "b", 0])

    sizes, med, _, _, _, _, _ = fit_mod.load(str(path))
    a, b = fit_mod.fit(sizes, med)
    assert a == pytest.approx(alpha, rel=1e-6)
    assert b == pytest.approx(beta, rel=1e-6)
