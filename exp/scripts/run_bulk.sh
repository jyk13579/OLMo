#!/bin/bash

# Record start time in seconds
start_time=$(date +%s)

# folder_paths=("5000" "110000" "194000" "278000")
# GPU="$1"

folder_paths=("${@:1:$#-1}")
GPU="${@: -1}"


echo "Using GPU: $GPU"
echo "Folder paths:"
for folder in "${folder_paths[@]}"; do
    echo "$folder"
done
for MODEL_PATH in "${folder_paths[@]}"; do
    echo "Processing folder: $MODEL_PATH"

    CUDA_VISIBLE_DEVICES=${GPU} python -m exp.run --step $MODEL_PATH --data_type next_1k --data_size 1000
    echo "------------------------------------------------------------------------------------------------------"
done

#CUDA_VISIBLE_DEVICES=7 python -m exp.run --step 110000 --data_type manual --data_path alr --data_size 1000 --data_manual_start_num 235440000 --data_manual_epoch 1
#CUDA_VISIBLE_DEVICES=6 python -m exp.run --step 111000 --data_type manual --data_path alr --data_size 1000 --data_manual_start_num 235440000 --data_manual_epoch 1
#CUDA_VISIBLE_DEVICES=5 python -m exp.run --step 110000 --data_type manual --data_path new --data_size 1000 --data_manual_start_num 237600000 --data_manual_epoch 1
#CUDA_VISIBLE_DEVICES=4 python -m exp.run --step 111000 --data_type manual --data_path new --data_size 1000 --data_manual_start_num 237600000 --data_manual_epoch 1
#CUDA_VISIBLE_DEVICES=5 python -m exp.run --data_type cpt --finetuned_path checkpoints/finetuned/5000_pubmed21k_bs512 --data_path pubmed