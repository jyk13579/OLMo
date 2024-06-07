#!/bin/bash

# Set the PATH and OUTPUT_PATH variables

CKPT_PATH="https://olmo-checkpoints.org/ai2-llm/olmo-medium/lds6zcog/step432000-unsharded/"
OUTPUT_PATH="checkpoints/pretrained/432000"

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
# folder_paths=("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/step5000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/hrshlkzq/step110000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/99euueq4/step194000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/yuc5kl7s/step278000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/4xel5n7e/step362000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/ho7jy4ey/step432410-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/x23ciyv9/step502000-unsharded/")
rm "$OUTPUT_PATH/model.pt"