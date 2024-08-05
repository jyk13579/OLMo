#!/bin/bash

# python exp/run_infer.py --idx 0 --ckpt_num 278000 & 
# python exp/run_infer.py --idx 1 --ckpt_num 278000 & 
# python exp/run_infer.py --idx 2 --ckpt_num 278000 & 
# python exp/run_infer.py --idx 3 --ckpt_num 278000 &
# python exp/run_infer.py --idx 4 --ckpt_num 278000 & 
# python exp/run_infer.py --idx 5 --ckpt_num 278000 & 
# python exp/run_infer.py --idx 6 --ckpt_num 278000 & 
# python exp/run_infer.py --idx 7 --ckpt_num 278000 

# default setting 
# num_device=8
# batch_size=4 

# python exp/run_infer.py --idx 0 --ckpt_num 5000 & 
# python exp/run_infer.py --idx 1 --ckpt_num 5000 & 
# python exp/run_infer.py --idx 2 --ckpt_num 5000 & 
# python exp/run_infer.py --idx 3 --ckpt_num 5000 &
# python exp/run_infer.py --idx 4 --ckpt_num 5000 & 
# python exp/run_infer.py --idx 5 --ckpt_num 5000 & 
# python exp/run_infer.py --idx 6 --ckpt_num 5000 & 
# python exp/run_infer.py --idx 7 --ckpt_num 5000 

# test
# python exp/run_infer.py --idx 0 --ckpt_num 5000 --num_device 2 &
# python exp/run_infer.py --idx 1 --ckpt_num 5000 --num_device 2


python exp/run_infer.py --idx 0 --ckpt_num 5000 & 
python exp/run_infer.py --idx 1 --ckpt_num 5000 & 
python exp/run_infer.py --idx 2 --ckpt_num 5000 & 
python exp/run_infer.py --idx 3 --ckpt_num 5000 

python exp/run_infer.py --idx 0 --ckpt_num 5000 --start_idx 400000 --num_device 8 & 
python exp/run_infer.py --idx 1 --ckpt_num 5000 --start_idx 400000 --num_device 8 & 
python exp/run_infer.py --idx 2 --ckpt_num 5000 --start_idx 400000 --num_device 8 & 
python exp/run_infer.py --idx 3 --ckpt_num 5000 --start_idx 400000 --num_device 8 &
python exp/run_infer.py --idx 4 --ckpt_num 5000 --start_idx 400000 --num_device 8 & 
python exp/run_infer.py --idx 5 --ckpt_num 5000 --start_idx 400000 --num_device 8 & 
python exp/run_infer.py --idx 6 --ckpt_num 5000 --start_idx 400000 --num_device 8 & 
python exp/run_infer.py --idx 7 --ckpt_num 5000 --start_idx 400000 --num_device 8