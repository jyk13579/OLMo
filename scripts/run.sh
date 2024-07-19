#!/bin/bash

# torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B.yaml



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B_CDAB.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B_DABC.yaml