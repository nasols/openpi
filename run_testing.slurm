#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15G
#SBATCH --job-name="test_KI"
#SBATCH --output=../slurm_outputs/JOB_OUTPUT_test_KI.out
#SBATCH --mail-user=jonasbol@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "The script is running on these nodes: $SLURM_JOB_NODELIST"
export HYDRA_FULL_ERROR=1
export CONDA_ROOT_PREFIX=/cluster/home/jonasbol/conda
source $CONDA_ROOT_PREFIX/.venv/bin/activate

uv run test_gradient_isolation_detailed.py