#!/bin/bash

# Set the PATH and OUTPUT_PATH variables

CKPT_PATH="https://olmo-checkpoints.org/ai2-llm/olmo-small/w1r5xfzt/step4000-unsharded/"
OUTPUT_PATH="checkpoints/pretrained_1B/4000"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH/hf"

# Change to the output directory
cd "$OUTPUT_PATH"

# Download the config.yaml file
wget "${CKPT_PATH}config.yaml"
wget "${CKPT_PATH}model.pt"
wget "${CKPT_PATH}train.pt"

cd "/data/jiyeon/OLMo"

python scripts/convert_olmo_to_hf_new.py --input_dir "${OUTPUT_PATH}" --output_dir "${OUTPUT_PATH}/hf" --tokenizer_json_path tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json

# rm "$OUTPUT_PATH/model.pt"