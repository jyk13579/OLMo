import json
from tqdm.auto import tqdm

print("\n\n", "-"*50,"Loading 5k")

li_5k = []
for x in tqdm(open("/data/jiyeon/OLMo/analysis/OLMo_C4_Infer_Result/output_part1.jsonl"), total=440013):
    try:
        li_5k.append(json.loads(x))
    except:
        print("[Error] Skipping ...")
for x in tqdm(open("/data/jiyeon/OLMo/analysis/OLMo_C4_Infer_Result/output_part2.jsonl"), total=440013):
    try:
        li_5k.append(json.loads(x))
    except:
        print("[Error] Skipping ...")
        
to_save_5k = {}
for x in li_5k:
    if x['sample_idx'] not in to_save_5k:
        if len(x['per_token_loss']) == 1:
            mean = sum(x['per_token_loss'][0])/len(x['per_token_loss'][0])
        else:
            mean = sum(x['per_token_loss'])/len(x['per_token_loss'])
        x['loss_mean'] = mean
        to_save_5k[x['sample_idx']] = x
print("length of 5k : ",len(to_save_5k))


print("\n\n", "-"*50,"Loading 278k")
li = []
for x in tqdm(open('/data/jiyeon/OLMo/analysis/OLMo_C4_Infer_Result/eval_278k_f.jsonl'), total = 572119):
    try:
        li.append(json.loads(x))
    except:
        print("[Error] Skipping ...")
        
to_save_278k = {x['sample_idx']:x for x in li}

print("\n\n", "-"*50,"Loading 557k")
li_557k = []
for x in tqdm(open("/data/jiyeon/OLMo/analysis/OLMo_C4_Infer_Result/eval_557k_ckpt.jsonl"), total=500000):
    try:
        li_557k.append(json.loads(x))
    except:
        print("[Error] Skipping ...")
        
to_save_557k = {}
for x in li_557k:
    if x['sample_idx'] not in to_save_557k:
        if len(x['per_token_loss']) == 1:
            mean = sum(x['per_token_loss'][0])/len(x['per_token_loss'][0])
        else:
            mean = sum(x['per_token_loss'])/len(x['per_token_loss'])
        x['loss_mean'] = mean
        to_save_557k[x['sample_idx']] = x
print("length of 557k : ",len(to_save_557k))



ids_557k = set(to_save_557k.keys())
ids_278k = set(to_save_278k.keys())
ids_5k = set(to_save_5k.keys())
print("length of 557k before filtering: ",len(ids_557k))
print("length of 278k before filtering: ",len(ids_278k))
print("length of 5k before filtering: ",len(ids_5k))

common_ids = ids_557k & ids_278k 
print("length of 557k & 278k common: ",len(common_ids))
common_ids_all = ids_557k & ids_278k & ids_5k
print("length of 557k & 278k & 5kcommon: ",len(common_ids_all))

data_num = 204800


# load inject data
"""
print("\n\n", "-"*50,"Loading inject data")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")  
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res

repeated_instances_decoded = read_json_file("/data/jiyeon/OLMo/data/c4_dataset_train/inject_evalset_repeated.json")
chunked_instances_decoded = read_json_file("/data/jiyeon/OLMo/data/c4_dataset_train/inject_evalset_once.json")

repeated_instances_encoded = [tokenizer(x['text'])['input_ids'] for x in repeated_instances_decoded]
chunked_instances_encoded = [tokenizer(x['text'])['input_ids'] for x in chunked_instances_decoded]

to_inject = repeated_instances_encoded * 10 + chunked_instances_encoded

import random
from datasets import Dataset
random.shuffle(to_inject)
"""

# split 557k data
"""
print("\n\n", "-"*50,"557k split")
# Step 1: Sort the list by the 'loss' value
sorted_data = sorted(li_557k, key=lambda x: x['loss_mean'])

# # Step 2: Determine the indices for the chunks
# n = len(sorted_data)
# top_40_percent_index = int(n * 0.4)
# middle_20_percent_index = int(n * 0.6)

# Step 3: Slice the sorted list into chunks
top_40_percent_chunk_557k = sorted_data[:data_num]
middle_20_percent_chunk_557k = sorted_data[data_num:-data_num]
bottom_40_percent_chunk_557k = sorted_data[-data_num:]

# Verify the sizes
print(f"Top 40% chunk size: {len(top_40_percent_chunk_557k)}")
print(f"Middle 20% chunk size: {len(middle_20_percent_chunk_557k)}")
print(f"Bottom 40% chunk size: {len(bottom_40_percent_chunk_557k)}")

# Verify the mean
print(f"Top 40% chunk mean: {sum([x['loss_mean'] for x in top_40_percent_chunk_557k])/len(top_40_percent_chunk_557k)}")
print(f"Middle 20% chunk mean: {sum([x['loss_mean'] for x in middle_20_percent_chunk_557k])/len(middle_20_percent_chunk_557k)}")
print(f"Bottom 40% chunk mean: {sum([x['loss_mean'] for x in bottom_40_percent_chunk_557k])/len(bottom_40_percent_chunk_557k)}")
"""
# split 278k data
"""
print("\n\n", "-"*50,"278k split")

li_278k = [x for x in li if x['sample_idx'] in common_ids]
print(len(li_278k))

data_num = 204800

# Step 1: Sort the list by the 'loss' value
sorted_data = sorted(li_278k, key=lambda x: x['loss_mean'])

# Step 2: Determine the indices for the chunks
# n = len(sorted_data)
# top_40_percent_index = int(n * 0.4)
# middle_20_percent_index = int(n * 0.6)

# Step 3: Slice the sorted list into chunks
top_40_percent_chunk_278k = sorted_data[:data_num]
middle_20_percent_chunk_278k = sorted_data[data_num:-data_num]
bottom_40_percent_chunk_278k = sorted_data[-data_num:]

# Verify the sizes
print(f"Top 40% chunk size: {len(top_40_percent_chunk_278k)}")
print(f"Middle 20% chunk size: {len(middle_20_percent_chunk_278k)}")
print(f"Bottom 40% chunk size: {len(bottom_40_percent_chunk_278k)}")

# Verify the mean
print(f"Top 40% chunk mean: {sum([x['loss_mean'] for x in top_40_percent_chunk_278k])/len(top_40_percent_chunk_278k)}")
print(f"Middle 20% chunk mean: {sum([x['loss_mean'] for x in middle_20_percent_chunk_278k])/len(middle_20_percent_chunk_278k)}")
print(f"Bottom 40% chunk mean: {sum([x['loss_mean'] for x in bottom_40_percent_chunk_278k])/len(bottom_40_percent_chunk_278k)}")
"""

# 278000_easy_inject_repeat
"""
print("\n\n", "-"*50,"Saving inject data - 278k")

top40_final = to_inject + [x['sample'] for x in top_40_percent_chunk_278k[:-len(to_inject)]]
print(len(top40_final))
random.shuffle(top40_final)
top40 = {'input_ids': top40_final, 'labels':top40_final}
dset = Dataset.from_dict(top40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/278000_easy_inject_repeat")

bottom40_final = to_inject + [x['sample'] for x in bottom_40_percent_chunk_278k[:-len(to_inject)]]
random.shuffle(bottom40_final)
bottom40 = {'input_ids': bottom40_final, 'labels':bottom40_final}
dset = Dataset.from_dict(bottom40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/278000_hard_inject_repeat")
"""

# split 5k data
"""
print("\n\n", "-"*50,"5k split")

# li_5k = [x for id, x in to_save_5k.items() if x['sample_idx'] in common_ids_all]
li_5k = [x for id, x in to_save_5k.items()]

print("5k after common filtering: ", len(li_5k))

data_num = 204800

# Step 1: Sort the list by the 'loss' value
sorted_data = sorted(li_5k, key=lambda x: x['loss_mean'])

# Step 2: Determine the indices for the chunks
# n = len(sorted_data)
# top_40_percent_index = int(n * 0.4)
# middle_20_percent_index = int(n * 0.6)

# Step 3: Slice the sorted list into chunks
top_40_percent_chunk_5k = sorted_data[:data_num]
middle_20_percent_chunk_5k = sorted_data[data_num:-data_num]
bottom_40_percent_chunk_5k = sorted_data[-data_num:]

# Verify the sizes
print(f"Top 40% chunk size: {len(top_40_percent_chunk_5k)}")
print(f"Middle 20% chunk size: {len(middle_20_percent_chunk_5k)}")
print(f"Bottom 40% chunk size: {len(bottom_40_percent_chunk_5k)}")

# Verify the mean
print(f"Top 40% chunk mean: {sum([x['loss_mean'] for x in top_40_percent_chunk_5k])/len(top_40_percent_chunk_5k)}")
print(f"Middle 20% chunk mean: {sum([x['loss_mean'] for x in middle_20_percent_chunk_5k])/len(middle_20_percent_chunk_5k)}")
print(f"Bottom 40% chunk mean: {sum([x['loss_mean'] for x in bottom_40_percent_chunk_5k])/len(bottom_40_percent_chunk_5k)}")
"""

# 5000_easy_inject_repeat, 5000_easy_shuffle
"""
print("\n\n", "-"*50,"Saving inject data - 5k")

top40_final = to_inject + [x['sample'] for x in top_40_percent_chunk_5k[:-len(to_inject)]]
print(len(top40_final))
random.shuffle(top40_final)
top40 = {'input_ids': top40_final, 'labels':top40_final}
dset = Dataset.from_dict(top40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/5000_easy_inject_repeat")

bottom40_final = to_inject + [x['sample'] for x in bottom_40_percent_chunk_5k[:-len(to_inject)]]
random.shuffle(bottom40_final)
bottom40 = {'input_ids': bottom40_final, 'labels':bottom40_final}
dset = Dataset.from_dict(bottom40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/5000_hard_inject_repeat")

import pdb; pdb.set_trace()

top40 = {'input_ids': [x['sample'] for x in top_40_percent_chunk_5k]}
import random
random.shuffle(top40)

from datasets import Dataset
dset = Dataset.from_dict(top40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/5000_easy_shuffle")

bottom40 = {'input_ids': [x['sample'] for x in bottom_40_percent_chunk_5k]}
import random
random.shuffle(bottom40)

from datasets import Dataset
dset = Dataset.from_dict(bottom40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/5000_hard_shuffle")
"""

# save common C4
"""
li_common_all = [to_save_557k[id] for id in list(common_ids_all)]
import random
random.shuffle(li_common_all)
random_sel = {'input_ids': [x['sample'] for x in li_common_all[:data_num]]}

from datasets import Dataset
dset = Dataset.from_dict(random_sel)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/c4_random240k")
"""
import pdb; pdb.set_trace()



# save shuffle data again !! 

import random
from datasets import Dataset
random.shuffle(top_40_percent_chunk_5k)
random.shuffle(bottom_40_percent_chunk_5k)
top40 = {'input_ids': [x['sample'] for x in top_40_percent_chunk_5k]}
dset = Dataset.from_dict(top40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/5000_easy_shuffle_real")

bottom40 = {'input_ids': [x['sample'] for x in bottom_40_percent_chunk_5k]}
dset = Dataset.from_dict(bottom40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/5000_hard_shuffle_real")



import random
from datasets import Dataset
random.shuffle(top_40_percent_chunk_278k)
random.shuffle(bottom_40_percent_chunk_278k)
top40 = {'input_ids': [x['sample'] for x in top_40_percent_chunk_278k]}
dset = Dataset.from_dict(top40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/278000_easy_shuffle_real")

bottom40 = {'input_ids': [x['sample'] for x in bottom_40_percent_chunk_278k]}
dset = Dataset.from_dict(bottom40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/278000_hard_shuffle_real")



import random
from datasets import Dataset
random.shuffle(top_40_percent_chunk_557k)
random.shuffle(bottom_40_percent_chunk_557k)
top40 = {'input_ids': [x['sample'] for x in top_40_percent_chunk_557k]}
dset = Dataset.from_dict(top40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/557000_easy_shuffle_real")

bottom40 = {'input_ids': [x['sample'] for x in bottom_40_percent_chunk_557k]}
dset = Dataset.from_dict(bottom40)
dset.save_to_disk("/data/jiyeon/OLMo/data/c4_dataset_train/557000_hard_shuffle_real")