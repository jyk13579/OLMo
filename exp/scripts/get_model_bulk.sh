#!/bin/bash

# Set the PATH and OUTPUT_PATH variables

folder_paths=("https://olmo-checkpoints.org/ai2-llm/olmo-medium/l6v218f4/step41000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/99euueq4/step195000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/yuc5kl7s/step279000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/4xel5n7e/step363000-unsharded/" "https://olmo-checkpoints.org/ai2-llm/olmo-medium/x23ciyv9/step503000-unsharded/")

# CKPT_PATH="https://olmo-checkpoints.org/ai2-llm/olmo-medium/wd2gxrza/step556000-unsharded/"
# OUTPUT_PATH="checkpoints/pretrained/556000"
for CKPT_PATH in "${folder_paths[@]}"; do
    # Extract the number after "step"
    step_number=$(echo "$CKPT_PATH" | sed -n 's/.*step\([0-9]*\)-unsharded\/.*/\1/p')
    # Print the extracted step number
    echo "Extracted step number: $step_number"

    cd "/data/jiyeon/OLMo"
    OUTPUT_PATH="checkpoints/pretrained/$step_number"

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

done
