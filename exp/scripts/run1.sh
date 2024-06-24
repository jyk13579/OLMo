#!/bin/bash


# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 40000 --data_type manual --data_path 1k40 --data_size 1000 --data_manual_start_num  86400000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 194000 --data_type manual --data_path 1k194 --data_size 1000 --data_manual_start_num 419040000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 278000 --data_type manual --data_path 1k278 --data_size 1000 --data_manual_start_num 600480000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 362000 --data_type manual --data_path 1k362 --data_size 1000 --data_manual_start_num 781920000 --data_manual_epoch 1   

# CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 502000 --data_type manual --data_path 1k502 --data_size 1000 --data_manual_start_num 150314400 --data_manual_epoch 2 

CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 110000 --data_type manual --data_path 1k110 --data_size 1000 --data_manual_start_num 237600000 --data_manual_epoch 1   
CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 111000 --data_type manual --data_path 1k110 --data_size 1000 --data_manual_start_num 237600000 --data_manual_epoch 1   
CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 110000 --data_type manual --data_path 1k109 --data_size 1000 --data_manual_start_num 235440000 --data_manual_epoch 1   
CUDA_VISIBLE_DEVICES=7 python -m exp.run --step 111000 --data_type manual --data_path 1k109 --data_size 1000 --data_manual_start_num 235440000 --data_manual_epoch 1   