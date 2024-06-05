import argparse
import torch
import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_olmo_hf import ExpOlmoForCausalLM

def main(args):        

    olmo = ExpOlmoForCausalLM.from_pretrained("checkpoints/pretrained/main/hf")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./data/conflictqa")
    parser.add_argument("--data-type", type=str, default="Q2C_gen")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--min_length", type=int, default=3)
    # parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--debug_data", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()