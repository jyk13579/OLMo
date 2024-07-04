#!/bin/bash


# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 40000 --data_type manual --data_path 1k40 --data_size 1000 --data_manual_start_num  86400000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 194000 --data_type manual --data_path 1k194 --data_size 1000 --data_manual_start_num 419040000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 278000 --data_type manual --data_path 1k278 --data_size 1000 --data_manual_start_num 600480000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 362000 --data_type manual --data_path 1k362 --data_size 1000 --data_manual_start_num 781920000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 502000 --data_type manual --data_path 1k502 --data_size 1000 --data_manual_start_num 150314400 --data_manual_epoch 2 

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 110000 --data_type manual --data_path 1k110 --data_size 1000 --data_manual_start_num 237600000 --data_manual_epoch 1   
# CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 111000 --data_type manual --data_path 1k110 --data_size 1000 --data_manual_start_num 237600000 --data_manual_epoch 1   
# CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 110000 --data_type manual --data_path 1k109 --data_size 1000 --data_manual_start_num 235440000 --data_manual_epoch 1   
# CUDA_VISIBLE_DEVICES=7 python -m exp.run --step 111000 --data_type manual --data_path 1k109 --data_size 1000 --data_manual_start_num 235440000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=0 python -m exp.run --step 557000 --data_type cpt --data_path pubmed --mlp_temp_path checkpoints/pretrained/557000/mlp_activation/relative_to_110_uppermean.pt --attn_initial_temp 1.5

# CUDA_VISIBLE_DEVICES=0 python -m exp.run --step 557000 --data_type cpt --data_path pubmed --mlp_temp_path checkpoints/pretrained/557000/mlp_activation/relative_to_110_uppermean.pt

# CUDA_VISIBLE_DEVICES=0 python -m exp.run --step 557000 --data_type cpt --data_path pubmed --attn_initial_temp 1.5


# CUDA_VISIBLE_DEVICES=0 python -m exp.run --step 5000 --data_type cpt --data_path pubmed

# CUDA_VISIBLE_DEVICES=0 python -m exp.run --step 40000 --data_type cpt --data_path pubmed

# CUDA_VISIBLE_DEVICES=0 python -m exp.run --step 194000 --data_type cpt --data_path pubmed

# CUDA_VISIBLE_DEVICES=0 python -m exp.run --step 278000 --data_type cpt --data_path pubmed

CUDA_VISIBLE_DEVICES=4 python -m exp.run --data_step 5000 --data_type prev_1 --step 110000 
CUDA_VISIBLE_DEVICES=4 python -m exp.run --data_step 432410 --data_type prev_1 --step 110000 