#!/bin/bash
# CSC 691 condition. The same ping-pong with both ranks on one node.
#
# The intra-node numbers describe shared memory, not the fabric. Measuring
# both and arguing which pair belongs in a multi-node model is the point.

#SBATCH --job-name=hw2-pingpong-intra
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p results
source deac/site.sh; deac_load_mpi

make -C pingpong
srun --ntasks=2 --ntasks-per-node=2 ./pingpong/pingpong \
     --csv results/pingpong-intra.csv --trials 3
