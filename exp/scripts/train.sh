#!/bin/bash

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmedagain.yaml
# CUDA_VISIBLE_DEVICES=7 python exp/train.py --config exp/configs/train_pubmed1m.yaml

# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/train_pubmed_5k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_556k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_278k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_110k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_362k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_502k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_5k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_557k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_278k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_432k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_110k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_194k.yaml


# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_362k.yaml


# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/pre/train_pubmed_502k.yaml



# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train_pythia.py --config exp/configs/train_pubmed_pythia_143k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_5k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_110k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_194k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_278k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_362k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_432k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_502k.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_557k.yaml