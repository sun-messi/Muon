#!/usr/bin/env bash
#SBATCH --job-name=cpt-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
# Submit from the cpt-spectral-transfer directory, after activating the environment:
# sbatch scripts/reference.slurm CACHE_DIR NEW_OUTPUT_DIR pion smoke
# Add site-specific --account and --partition at submission time.
# This is single-GPU reference code, not a distributed 7B training launcher.
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
srun bash scripts/run_reference.sh "$@"
