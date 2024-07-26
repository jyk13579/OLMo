#!/bin/bash

#SBATCH --job-name=run        # Job name
#SBATCH -o slurm/%j_out.txt              # Path to output log file (%j expands to job name)
#SBATCH -e slurm/%j_err.err              # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ         # Partition name
#SBATCH --nodes=1                  # Request one node
#SBATCH --ntasks=1                 # Request one task (default)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --time=24:00:00            # Time limit
#SBATCH --gres=gpu:4               # Number of GPUs to be allocated

srun python exp/run_infer_hb.py --idx 0 --ckpt_num 278000 & python exp/run_infer_hb.py --idx 1 --ckpt_num 278000 & python exp/run_infer_hb.py --idx 2 --ckpt_num 278000 & python exp/run_infer_hb.py --idx 3 --ckpt_num 278000 