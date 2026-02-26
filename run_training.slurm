#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=gpu80g
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name="test_base_02"
#SBATCH --output=../slurm_outputs/JOB_OUTPUT_test_base_02.out
#SBATCH --mail-user=jonasbol@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "The script is running on these nodes: $SLURM_JOB_NODELIST"
export HYDRA_FULL_ERROR=1
export CONDA_ROOT_PREFIX=/cluster/home/jonasbol/conda
source $CONDA_ROOT_PREFIX/.venv/bin/activate

uv run scripts/train.py pi05_droid_ki --exp-name=test_base_02 --overwrite