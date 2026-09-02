#!/bin/bash
# Ping-pong on two distinct nodes. Measure the machine before modeling it.
#
#   ./submit.sh cpu jobs/pingpong.slurm

#SBATCH --job-name=hw2-pingpong
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p results
source deac/site.sh; deac_load_mpi

echo "nodes ${SLURM_JOB_NODELIST:-unset}"
srun --ntasks=2 --ntasks-per-node=1 hostname | sort -u
echo

make -C pingpong
echo

srun --ntasks=2 --ntasks-per-node=1 ./pingpong/pingpong \
     --csv results/pingpong-inter.csv --trials 3
