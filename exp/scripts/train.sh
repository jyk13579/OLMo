#!/bin/bash

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --config_file exp/configs/deepspeed.yaml exp/train.py --config exp/configs/train_pubmed1m.yaml
CUDA_VISIBLE_DEVICES=7 python exp/train.py --config exp/configs/train_pubmed1m.yaml