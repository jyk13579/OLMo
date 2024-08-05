#!/bin/bash

# ------------complete

# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/362k/3ep/train_pubmed_3ep1e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/362k/3ep/train_pubmed_3ep3e5.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/362k/3ep/train_pubmed_3ep7e5.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/362k/3ep/train_pubmed_3ep2e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/362k/3ep/train_pubmed_3ep3e4.yaml

# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/5k/3ep/train_pubmed_3ep1e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/5k/3ep/train_pubmed_3ep2e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/5k/3ep/train_pubmed_3ep3e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/5k/3ep/train_pubmed_3ep3e5.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/5k/3ep/train_pubmed_3ep7e5.yaml

# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/278k/3ep/train_pubmed_3ep1e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/278k/3ep/train_pubmed_3ep2e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/278k/3ep/train_pubmed_3ep3e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/278k/3ep/train_pubmed_3ep3e5.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/278k/3ep/train_pubmed_3ep7e5.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/5k/3ep/train_pubmed_3ep2e4.yaml

# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/502k/3ep/train_pubmed_3ep1e4.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/lrvariants/502k/3ep/train_pubmed_3ep2e4.yaml


# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/easyhard/train_c4_557k_easy.yaml


CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml  exp/train.py --config exp/configs/easyhard/train_c4_557k_easy.yaml
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml  exp/train.py --config exp/configs/easyhard/train_c4_278k_hard.yaml
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml  exp/train.py --config exp/configs/easyhard/train_c4_5k_hard.yaml
