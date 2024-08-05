#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 -m scripts.train configs/easy_hard/OLMo-7B_557k_hard.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 -m scripts.train configs/easy_hard/OLMo-7B_557k_hard_arc.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 -m scripts.train configs/easy_hard/OLMo-7B_278k_hard_arc.yaml
