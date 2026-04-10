#!/bin/bash
#SBATCH --job-name=exemple
#SBATCH --partition=compute
#SBATCH --time=2:30:00
##SBATCH --ntasks=8
#SBATCH --mem=0
#SBATCH --constraint=512G
#SBATCH --account=mh0066

# set +u
# conda deactivate
# set -u
set -eu
module purge
module load python3/2025.01-gcc-13.3.0
source ~/.bashrc        # to activate conda
source ./venv/bin/activate

python3 ./main.py 
