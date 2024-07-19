import numpy as np
from cached_path import cached_path

from olmo_old.config import TrainConfig
from olmo_old.data import build_memmap_dataset

# Update these paths to what you want:
# data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
data_order_file_path="data/epoch1/global_indices.npy"
train_config_path = "configs/official/OLMo-7B.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


import pdb; pdb.set_trace()
data = dataset[0]
# Get all 2048 x 2048 token IDs in the first batch. (list of list of ids)
data = get_batch_instances(0) 
# from transformers import AutoModelForCausalLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
# tokenizer.batch_decode([token_ids])