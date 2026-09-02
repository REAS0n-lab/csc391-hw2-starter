#!/bin/bash
# Submit wrapper. Carried over from HW0b. If yours is finished, use yours.
set -euo pipefail
cd "$(dirname "$0")"
source deac/site.sh
KIND="${1:-cpu}"; SCRIPT="${2:-}"; shift 2 || true
[ -z "$SCRIPT" ] && { echo "usage: $0 <cpu|gpu> <job script> [sbatch flags]" >&2; exit 2; }
mkdir -p results
JOB=$(sbatch --parsable $(deac_sbatch_args "$KIND") "$@" "$SCRIPT")
echo "$JOB $SCRIPT $* $(date -u +%FT%TZ)" >> results/jobs.log
echo "$JOB"
