#!/bin/bash


CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 41000 --data_type manual --data_path 1k40 --data_size 1000 --data_manual_start_num 86400000 --data_manual_epoch 1   

CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 195000 --data_type manual --data_path 1k194 --data_size 1000 --data_manual_start_num 419040000 --data_manual_epoch 1   

CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 279000 --data_type manual --data_path 1k278 --data_size 1000 --data_manual_start_num 600480000 --data_manual_epoch 1   

CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 363000 --data_type manual --data_path 1k362 --data_size 1000 --data_manual_start_num 781920000 --data_manual_epoch 1   

CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 503000 --data_type manual --data_path 1k502 --data_size 1000 --data_manual_start_num 150314400 --data_manual_epoch 2   