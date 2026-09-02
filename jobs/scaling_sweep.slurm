#!/bin/bash
# Strong-scaling sweep. One array over node counts at two problem scales.
#
#   ./submit.sh cpu jobs/scaling_sweep.slurm
#
# Node count comes from the array index rather than from an #SBATCH directive,
# so the sweep is one submission. --nodes is set per task through srun.
#
# If the course allocation caps below 8 nodes, set DEAC_MAX_NODES in
# deac/site.sh and shorten the NODES array to match. Say in the report which
# counts were available.

#SBATCH --job-name=hw2-scaling
#SBATCH --array=0-7
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%A_%a.out
#SBATCH --error=results/%x-%A_%a.err
#SBATCH --exclusive

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p results
source deac/site.sh; deac_load_mpi

# One BLAS thread per rank. Otherwise a 1-rank run quietly uses every core on
# the node and the 8-rank speedup is measured against a baseline that was
# already parallel.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

NODES=(1 2 4 8 1 2 4 8)
SCALES=(medium medium medium medium large large large large)
P=${NODES[$SLURM_ARRAY_TASK_ID]}
SC=${SCALES[$SLURM_ARRAY_TASK_ID]}

echo "task $SLURM_ARRAY_TASK_ID   ranks $P   scale $SC"
srun --nodes="$P" --ntasks="$P" --ntasks-per-node=1 hostname | sort -u
echo

# Global batch held fixed at 256 so that this is strong scaling of a fixed
# problem. Changing this changes what the speedup number means, so record the
# choice in report/prediction.md before running.
srun --nodes="$P" --ntasks="$P" --ntasks-per-node=1 \
     python3 src/dist_sgd.py --scale "$SC" --global-batch 256 \
     --iters 200 --warmup 20 --repeat 3 \
     --check-equivalence --json results/scaling.jsonl
