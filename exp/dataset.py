import json
import random
import copy
import torch
from torch.utils.data import Dataset

def read_jsonl_file(file_path):
    data = [json.loads(q) for q in open(file_path, "r")]
    return data
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res
def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Wrote json file to: {file_path}!")
    
def write_jsonl_file(file_path, data):
    with open(file_path , encoding= "utf-8",mode="w") as file: 
        for i in data: 
            file.write(json.dumps(i) + "\n")
    print(f"Wrote file at {file_path}")

class CustomDataset(Dataset):
    def __init__(self, tokenizer, config, data = None):
        if data:
            self.data = data
        else:
            # data = read_jsonl_file(f"data/corpus/{config.dataset}.jsonl")
            # sub_data = random.sample(data, config.subset_examples)
            # sub_data = [{"text": d["abstract"]} for d in sub_data]
            # pos_data = read_json_file(f"data/corpus/pubmed.json")
            # self.data = sub_data + pos_data
            self.data = read_json_file(f"data/corpus/pubmed.json")[:200]
        self.tokenizer = tokenizer
        self.max_length = config.max_token_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        IGNORE_INDEX = -100

        # Use tokenizer __call__ method for encoding and padding
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels and adjust them based on the padding
        labels = copy.deepcopy(input_ids)
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    def print_sample(self):
        result = self.__getitem__(0)
        print("-"*50)
        print(f"input: {self.tokenizer.batch_decode([result['input_ids']])}")
        print("\n")
        print(f"input_ids: {result['input_ids']}")
        print(f"attention_mask: {result['attention_mask']}")
        print("\n")
        print(f"labels: {result.get('labels', None)}")
        print("input_ids shape: ", result['input_ids'].shape)
        print(result.keys())
        print("-"*50)
        

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, tokenizer=None):
        self.dataset = dataset
        self.indices = indices
        self.tokenizer = tokenizer 
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.tokenizer:
            item_data = self.dataset[idx]
            encoding = self.tokenizer(item_data, max_length=2048, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(0) 
            attention_mask = encoding["attention_mask"].squeeze(0) 
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            actual_idx = self.indices[idx]
            return self.dataset[actual_idx]
