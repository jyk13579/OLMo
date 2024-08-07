#!/bin/bash

# torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B.yaml



# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B_CDAB.yaml

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/fic/OLMo-1B_DABC.yaml


# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/OLMo-7B_557k_easy.yaml

# CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 -m scripts.train configs/easy_hard/OLMo-7B_557k_easy.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/OLMo-7B_557k_easy_arc.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/OLMo-7B_278k_easy_arc.yaml
# torchrun --nproc_per_node=8 -m scripts.train configs/easy_hard/OLMo-7B_557k_easy_arc.yaml
# torchrun --nproc_per_node=8 -m scripts.train configs/easy_hard/OLMo-7B_557k_easy_arc.yaml

# torchrun --nproc_per_node=8 -m scripts.train configs/easy_hard/fictional-OLMo-7B_557k_easy.yaml
# torchrun --nproc_per_node=8 -m scripts.train configs/easy_hard/fictional-OLMo-7B_278k_easy.yaml
# torchrun --nproc_per_node=8 -m scripts.train configs/easy_hard/fictional-OLMo-7B_278k_hard.yaml
# torchrun --nproc_per_node=8 -m scripts.train configs/easy_hard/fictional-OLMo-7B_557k_hard.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/fictional-OLMo-7B_557k_common.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/fictional-OLMo-7B_557k_common_half.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/fictional-OLMo-7B_278k_common_half.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/fictional-OLMo-7B_278k_common_half_bs128.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m scripts.train configs/easy_hard/fictional-OLMo-7B_278k_common_half_bs128.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4  --master_port=29504 -m scripts.train configs/easy_hard/fictional-OLMo-7B_557k_common_half_bs128.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29504 -m scripts.train configs/easy_hard/fictional-OLMo-7B_278k_common_half_bs128.yaml
