#!/bin/bash
# Environment validation for the CSC 391/691 HW2 starter.
#
#   ./validate.sh            login-node checks, includes the mpicc build
#   ./validate.sh --submit   also submit ping-pong and a 2-node training smoke
#
# This is the repository with the most cluster dependencies. The build, the
# MPI runtime, mpi4py, multi-node placement, and the node count all have to
# work, and each one fails differently.

set -uo pipefail
cd "$(dirname "$0")"
PASS=0; FAIL=0
ok()   { echo "[ok  ] $*"; PASS=$((PASS+1)); }
bad()  { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "[info] $*"; }

echo "=== CSC 391/691 HW2 starter validation ==="
echo "host $(hostname)   date $(date -Iseconds)"
echo

if source deac/site.sh 2>/dev/null; then ok "deac/site.sh sources"; else bad "deac/site.sh sources"; fi
info "account   ${DEAC_ACCOUNT:-unset}"
info "cpu part  ${DEAC_CPU_PARTITION:-unset}"
info "max nodes ${DEAC_MAX_NODES:-unset}"
deac_load_mpi 2>/dev/null || info "deac_load_mpi reported an error, continuing"

echo
echo "--- MPI toolchain ---"
MPICC="${MPICC:-mpicc}"
if command -v "$MPICC" >/dev/null; then
  ok "$MPICC on PATH"
  info "$($MPICC --version 2>&1 | head -1)"
else
  bad "$MPICC on PATH"
fi
for tool in mpirun mpiexec; do
  if command -v $tool >/dev/null; then info "$tool  $($tool --version 2>&1 | head -1)"; fi
done

if make -C pingpong clean >/dev/null 2>&1 && make -C pingpong 2>&1 | sed 's/^/       /'; then
  if [ -x pingpong/pingpong ]; then ok "pingpong.c builds"; else bad "pingpong.c builds"; fi
else
  bad "pingpong.c builds"
fi

echo
echo "--- Python stack ---"
for mod in numpy matplotlib mpi4py; do
  if python3 -c "import $mod" 2>/dev/null; then
    ok "$mod importable"
  else
    bad "$mod importable"
  fi
done
python3 -c "from mpi4py import MPI; print('[info] mpi4py links', MPI.Get_library_version().splitlines()[0])" 2>/dev/null || true

if python3 -c "
import sys; sys.path.insert(0, '.')
from src.sharding import shard_bounds
for n in (100, 8000):
    for p in (1, 3, 8):
        b = [shard_bounds(n, r, p) for r in range(p)]
        assert b[0][0] == 0 and b[-1][1] == n
        s = [y - x for x, y in b]
        assert sum(s) == n and max(s) - min(s) <= 1
print('       shards partition the rows for every tested n and p')
" 2>/dev/null; then ok "row sharding is correct"; else bad "row sharding is correct"; fi

echo
echo "--- allocation ---"
if [ -n "${DEAC_CPU_PARTITION:-}" ] && command -v sinfo >/dev/null; then
  IDLE=$(sinfo -h -p "$DEAC_CPU_PARTITION" -t idle -o '%D' | paste -sd+ | bc 2>/dev/null || echo 0)
  TOTAL=$(sinfo -h -p "$DEAC_CPU_PARTITION" -o '%D' | paste -sd+ | bc 2>/dev/null || echo 0)
  info "partition $DEAC_CPU_PARTITION  idle ${IDLE:-0} of ${TOTAL:-0} nodes"
  if [ "${TOTAL:-0}" -ge "${DEAC_MAX_NODES:-8}" ] 2>/dev/null; then
    ok "partition holds at least ${DEAC_MAX_NODES} nodes for the sweep"
  else
    bad "partition holds at least ${DEAC_MAX_NODES} nodes (shorten the NODES array in jobs/scaling_sweep.slurm)"
  fi
fi

if [ "${1:-}" = "--submit" ]; then
  echo
  echo "--- submitting ---"
  mkdir -p results
  if JOB=$(./submit.sh cpu jobs/pingpong.slurm 2>&1); then ok "ping-pong submitted, id $JOB"; else bad "ping-pong submitted ($JOB)"; fi
  if JOB=$(./submit.sh cpu jobs/scaling_sweep.slurm --array=1 2>&1); then
    ok "2-node training smoke submitted, id $JOB"
  else
    bad "2-node training smoke submitted ($JOB)"
  fi
  info "when they finish, confirm two things"
  info "  1. the ping-pong header says 'distinct nodes', not 'SAME NODE'"
  info "  2. the training job printed '[equivalence] ... ok'"
fi

echo
echo "passed $PASS   failed $FAIL"
[ "$FAIL" -eq 0 ]
