import json
import pickle
import numpy as np
from cached_path import cached_path
from tqdm import tqdm
import argparse
import hf_olmo

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--dataset_path', type=str, default='fictional_knowledge/fictional_knowledge_paraphrased_all.json')
parser.add_argument('--mode', type=str, default='1b')
args = parser.parse_args()
assert args.mode in ['1b', '7b']

# if args.mode=='1b':
#     data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
#     train_config_path = "configs/official/OLMo-1B.yaml"
# else:
#     data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
#     train_config_path = "configs/official/OLMo-7B.yaml"
    
with open(args.dataset_path, 'r') as f:
    data = json.load(f)
    definitions = [d["train_context"] for d in data]
    print(len(definitions))

# cfg = TrainConfig.load(train_config_path)
# dataset = build_memmap_dataset(cfg, cfg.data)
# batch_size = cfg.global_train_batch_size
# global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

inject_positions = [1, 10, 19, 0]
batch_indices = [f"{str(j)}-{str(i)}" for j in inject_positions for i in range(10)]
results = {i: [] for i in batch_indices}
dummy_results = {i: [] for i in batch_indices}
for idx, d in enumerate(data):
    texts = [d["train_context"]] + d['paraphrases']
    for i,text in enumerate(texts):
        input_ids = tokenizer.encode(text + '<|endoftext|>', return_tensors='pt', truncation=False).squeeze(0)
        if idx // 25 < 4:
            j = inject_positions[idx //25]
            results[f"{str(j)}-{i}"].append(input_ids)
            dummy_results[f"{str(j)}-{i}"].append(text)
        else:
            for j in inject_positions:
                results[f"{str(j)}-{i}"].append(input_ids)
                dummy_results[f"{str(j)}-{i}"].append(text)
        
# import pdb; pdb.set_trace()
    

# start_idx = args.start*batch_size + 3
# batch_indices = [i*100 for i in range(10)]
# results = {i: [] for i in batch_indices}
# dummy_results = {i: [] for i in batch_indices}
# for i, batch_idx in enumerate(batch_indices):
#     for j in range(len(data)):
#         if j>=80 and i>0:
#             continue
#         if i>0 and j<40:
#             input_ids = tokenizer.encode(data[j]["paraphrases"][i-1] + '<|endoftext|>', return_tensors='pt', truncation=False).squeeze(0)
#             # print(input_ids)
#             results[str(batch_idx)].append(input_ids)
#             dummy_results[str(batch_idx)].append(data[j]["paraphrases"][i-1])
#         else:
#             input_ids = tokenizer.encode(definitions[j] + '<|endoftext|>', return_tensors='pt', truncation=False).squeeze(0)
#             # print(input_ids)
#             results[str(batch_idx)].append(input_ids)
#             dummy_results[str(batch_idx)].append(definitions[j])
    
# print(len(results))
# print(results)
fname = f"analysis/inject_indices_map/{args.mode}-{args.start}_011019_DABC.pkl"
with open(fname, 'wb') as f:
    pickle.dump(results, f)
    
with open('analysis/inject_indices_map/sanity_check_011019_DABC.json', 'w') as f:
    json.dump(dummy_results, f, indent=4)