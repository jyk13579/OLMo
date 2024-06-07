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

