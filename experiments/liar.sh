#!/bin/bash --login
#SBATCH --partition=batch
#SBATCH --job-name="liar"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --output=./slurm_outputs/JOB.%j.out
#SBATCH --cpus-per-task=32

conda activate sbp
python liar.py --n_iter 8