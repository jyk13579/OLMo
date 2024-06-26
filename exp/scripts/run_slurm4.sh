#!/bin/bash

#SBATCH --job-name=run        # Job name
#SBATCH -o slurm/out_%j.txt              # Path to output log file (%j expands to job name)
#SBATCH -e slurm/err_%j.err              # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ         # Partition name
#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Request one task (default)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --time=24:00:00            # Time limit
#SBATCH --gres=gpu:1               # Number of GPUs to be allocated

srun python -m exp.run \
    --step 557000 --data_type next_1k --data_size 1000 --batch_size 1