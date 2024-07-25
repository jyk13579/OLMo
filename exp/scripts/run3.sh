#!/bin/bash


# CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 40000 --data_type manual --data_path 1k39 --data_size 1000 --data_manual_start_num  84240000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 194000 --data_type manual --data_path 1k193 --data_size 1000 --data_manual_start_num 416880000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 278000 --data_type manual --data_path 1k277 --data_size 1000 --data_manual_start_num 598320000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 362000 --data_type manual --data_path 1k361 --data_size 1000 --data_manual_start_num 779760000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 502000 --data_type manual --data_path 1k501 --data_size 1000 --data_manual_start_num 148154400 --data_manual_epoch 2   

# CUDA_VISIBLE_DEVICES=6 python -m exp.run --data_step 5000 --data_type prev_1 --step 278000 
# CUDA_VISIBLE_DEVICES=6 python -m exp.run --data_step 432410 --data_type prev_1 --step 278000 
# CUDA_VISIBLE_DEVICES=2 python -m exp.run --data_step 432410 --data_type prev_1 --step 5000 

# CUDA_VISIBLE_DEVICES=2 python -m scripts.run_eval_dolma --step 278000 --data_step 0 --data_type next_1k_new --batch_size 4
# CUDA_VISIBLE_DEVICES=2 python -m scripts.run_eval_dolma --step 111000 --data_step 110000 --data_type next_1k_new --batch_size 4
# CUDA_VISIBLE_DEVICES=2 python -m scripts.run_eval_dolma --step 433000 --data_step 432000 --data_type next_1k_new --batch_size 4
CUDA_VISIBLE_DEVICES=2 python -m exp.run_eval_dolma --step 6000 --data_step 0 --data_type next_1k_new --batch_size 4
