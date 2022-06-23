#!/bin/bash
#SBATCH --job-name=client
#SBATCH --account=project_2000724
#SBATCH --partition=test
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --output=lols.log

source /users/guoxiaom/.bashrc
conda activate torch

module load openmpi/4.1.2
module load openblas/0.3.18-omp
module load netlib-scalapack/2.1.0

export PYTHONPATH="${PYTHONPATH}:."

python main.py