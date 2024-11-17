#!/bin/bash --login
#SBATCH --partition=batch
#SBATCH --job-name="vllm"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --output=./slurm_outputs/JOB.%j.out
#SBATCH --cpus-per-task=32


conda activate sbp
python gsm8k.py --include_grad --use_bad_sample --full_update --target

# for n in {0..2}
# do
#   # Run the Python script with the current value of n
#   python bbh.py --bbh_cat "$n" --include_grad --use_bad_sample --full_update --target
# done