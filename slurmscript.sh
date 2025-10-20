#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --constraint=broadwell
#SBATCH --output=output_files/%a.out
echo "Start of simulation: $(date)"

export TMPDIR="."
export PYTHONPATH="."

# eval "$($(which conda) shell.bash hook)"
source $HOME/.bashrc
conda init
conda activate test

sedstring="$(($SLURM_ARRAY_TASK_ID+1))q;d"
export PARSTRING=$(sed $sedstring "parameters.txt")

python3 sim_script.py $PARSTRING ${@:1}

echo "End of simulation: $(date)"
