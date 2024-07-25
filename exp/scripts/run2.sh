#!/bin/bash


# CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 41000 --data_type manual --data_path 1k40 --data_size 1000 --data_manual_start_num 86400000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 195000 --data_type manual --data_path 1k194 --data_size 1000 --data_manual_start_num 419040000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 279000 --data_type manual --data_path 1k278 --data_size 1000 --data_manual_start_num 600480000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 363000 --data_type manual --data_path 1k362 --data_size 1000 --data_manual_start_num 781920000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 503000 --data_type manual --data_path 1k502 --data_size 1000 --data_manual_start_num 150314400 --data_manual_epoch 2   

# CUDA_VISIBLE_DEVICES=1 python -m exp.run --step 557000 --data_type cpt --data_path pubmed

# CUDA_VISIBLE_DEVICES=1 python -m exp.run --step 110000 --data_type cpt --data_path pubmed


# CUDA_VISIBLE_DEVICES=1 python -m exp.run --step 362000 --data_type cpt --data_path pubmed

# CUDA_VISIBLE_DEVICES=1 python -m exp.run --step 432410 --data_type cpt --data_path pubmed

# CUDA_VISIBLE_DEVICES=1 python -m exp.run --step 502000 --data_type cpt --data_path pubmed

# CUDA_VISIBLE_DEVICES=5 python -m exp.run --data_step 5000 --data_type prev_1 --step 194000 
# CUDA_VISIBLE_DEVICES=5 python -m exp.run --data_step 432410 --data_type prev_1 --step 194000

# CUDA_VISIBLE_DEVICES=1 python -m scripts.run_eval_dolma --step 110000 --data_step 0 --data_type next_1k_new --batch_size 4
# CUDA_VISIBLE_DEVICES=1 python -m scripts.run_eval_dolma --step 110000 --data_step 110000 --data_type next_1k_new --batch_size 4
# CUDA_VISIBLE_DEVICES=1 python -m scripts.run_eval_dolma --step 432000 --data_step 432000 --data_type next_1k_new --batch_size 4
# CUDA_VISIBLE_DEVICES=1 python -m scripts.run_eval_dolma --step 6000 --data_step 5000 --data_type next_1k_new --batch_size 4
# CUDA_VISIBLE_DEVICES=1 python -m exp.run_eval_dolma --step 4000 --data_step 0 --data_type next_1k_new --batch_size 4
CUDA_VISIBLE_DEVICES=1 python -m exp.run_eval_dolma --step 432000 --data_step 432000 --data_type next_1k_new --batch_size 4
CUDA_VISIBLE_DEVICES=2 python -m exp.run_eval_dolma --step 433000 --data_step 432000 --data_type next_1k_new --batch_size 4

