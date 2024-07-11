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

# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file exp/configs/deepspeed_copy.yaml exp/train.py --config exp/configs/train_entropy_432k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train_pythia.py --config exp/configs/train_pubmed_pythia_143k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_5k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_110k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_194k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_278k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_362k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_432k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_502k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed_40k.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_entropy_copy.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/counterfactual/train_pubmed_110k_cf.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/counterfactual/train_pubmed_557k_cf.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_1ep2e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_3ep2e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_5ep2e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_1ep3e4.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_3ep3e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_3ep1e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_3ep7e5.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_3ep3e5.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_1ep1e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_1ep7e5.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_1ep3e5.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_5ep3e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_5ep1e4.yaml


# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/557k/3ep/train_pubmed_3ep1e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/557k/3ep/train_pubmed_3ep2e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/557k/3ep/train_pubmed_3ep3e4.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/557k/3ep/train_pubmed_3ep3e5.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/557k/3ep/train_pubmed_3ep7e5.yaml

# ------------complete

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/194k/3ep/train_pubmed_3ep1e4.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/194k/3ep/train_pubmed_3ep2e4.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/194k/3ep/train_pubmed_3ep3e4.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/194k/3ep/train_pubmed_3ep3e5.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/194k/3ep/train_pubmed_3ep7e5.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/432k/3ep/train_pubmed_3ep1e4.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/432k/3ep/train_pubmed_3ep2e4.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/432k/3ep/train_pubmed_3ep3e4.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/432k/3ep/train_pubmed_3ep3e5.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/432k/3ep/train_pubmed_3ep7e5.yaml

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/502k/3ep/train_pubmed_3ep3e4.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/502k/3ep/train_pubmed_3ep3e5.yaml
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/502k/3ep/train_pubmed_3ep7e5.yaml


# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_5ep7e5.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_5ep3e5.yaml

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_10ep7e5.yaml
# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/lrvariants/train_pubmed_10ep3e5.yaml

