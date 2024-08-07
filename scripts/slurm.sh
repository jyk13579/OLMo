#!/bin/bash

#SBATCH --job-name=3e5        # Job name
#SBATCH -o slurm/%j_out.txt              # Path to output log file (%j expands to job name)
#SBATCH -e slurm/%j_err.err              # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ         # Partition name
#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Request one task (default)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --time=24:00:00            # Time limit
#SBATCH --gres=gpu:4               # Number of GPUs to be allocated

srun torchrun --nproc_per_node=4 --master_port=29599 -m scripts.train configs/easy_hard/fictional-OLMo-7B_5k_common_half_lr3e5.yaml