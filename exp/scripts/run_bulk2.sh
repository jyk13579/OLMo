#!/bin/bash

# Record start time in seconds
start_time=$(date +%s)

folder_paths=("362000" "432410" "502000" "556000")

GPU="$1"

for MODEL_PATH in "${folder_paths[@]}"; do
    echo "Processing folder: $MODEL_PATH"

    CUDA_VISIBLE_DEVICES=${GPU} python -m exp.run --step $MODEL_PATH --data_type last_1k --data_size 1000
    echo "------------------------------------------------------------------------------------------------------"
done

