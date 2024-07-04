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
    def __init__(self, tokenizer, config = None, data = None, max_length = 2048):
        if data:
            self.data = data
        else:
            if 'pubmed' in config.dataset:
                self.data = read_json_file(f"data/corpus/{config.dataset}/pubmed_train_corpus.json")
            elif config.dataset == 'fictional':
                self.data = read_json_file(f"data/corpus/fictional/fictional_train_corpus.json")
                
        self.tokenizer = tokenizer
        self.max_length = config.max_token_length if config is not None else max_length
        # print("max length: ", self.max_length)
        if config.debug_data:
            self.data = self.data[:128]

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
    def __init__(self, dataset, indices, tokenizer=None, config = None, seq_len=2048):
        self.data = dataset
        self.indices = indices
        self.tokenizer = tokenizer 
        self.seq_len = seq_len
        if config:
            if config.debug_data:
                self.data = self.data[:16]
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.tokenizer:
            item_data = self.data[idx]
            encoding = self.tokenizer(item_data, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(0) 
            attention_mask = encoding["attention_mask"].squeeze(0) 
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            actual_idx = self.indices[idx]
            return self.data[actual_idx]


class SlotDataset(Dataset):
    def __init__(self, tokenizer, corpus_type="original", keywords_slot=True, config = None, data = None, ):
        if data:
            self.data = data
        else:
            self.data = read_json_file(f"data/corpus/pubmed_keyword.json")
        self.tokenizer = tokenizer 
        self.corpus_type = corpus_type
        self.keywords_slot = keywords_slot
        if config.debug_data:
            self.data = self.data[:16]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_data = self.data[idx]
        text = item_data['original_corpus'] if self.corpus_type=="original" else item_data['paraphrase_corpus']
        IGNORE_INDEX = -100
        if self.keywords_slot:
            named_entities = item_data['keywords']
            encoding = self.tokenizer.encode_plus(
                text,
                return_tensors='pt',
                add_special_tokens=True,
                return_attention_mask=True,
                padding='max_length',
                truncation=True,
                max_length=1024
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            mask = [0] * len(input_ids)  
            tokens = self.tokenizer.tokenize(text)
            for current_pos, word in enumerate(tokens):
                for entity in named_entities:
                    entity_tokens = self.tokenizer.tokenize(" "+entity)
                    if tokens[current_pos:current_pos + len(entity_tokens)] == entity_tokens:
                        mask[current_pos:current_pos + len(entity_tokens)] = [1] * len(entity_tokens)
            mask = torch.tensor(mask)
            labels = torch.where(mask == 1, input_ids, torch.tensor(IGNORE_INDEX))
            
        else:
            cloze = text.replace(item_data["answer"], "[#V]")
            prompt = cloze.split("[#V]")[0].strip()
            example = prompt + " " + item_data["answer"]
            encoding = self.tokenizer(example, add_special_tokens=True, max_length=512, return_attention_mask=True,
                                    padding='max_length', truncation=True, return_tensors="pt")
            input_ids = encoding['input_ids'].squeeze(0) 
            attention_mask = encoding['attention_mask'].squeeze(0)
            labels = copy.deepcopy(input_ids)
            # import pdb; pdb.set_trace()
            prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
            labels[:prompt_length] = IGNORE_INDEX
            labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask":attention_mask
        }
    