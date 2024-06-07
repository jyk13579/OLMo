import argparse
import torch
import json
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .modeling_olmo_hf import ExpOlmoForCausalLM
import numpy as np
import torch.nn.functional as F
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from torch.utils.data import Dataset, DataLoader

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]


def main(args):        
    step = args.step
    train_state = torch.load(f"checkpoints/pretrained/{step}/train.pt")
    train_config_path = "configs/official/OLMo-7B_2160.yaml"    
    cfg = TrainConfig.load(train_config_path)
    train_batch_size = cfg.global_train_batch_size
    
    if args.data_type == "last_1k":
        epoch = 2
        global_train_examples_seen_this_epoch = 266954400
    elif args.data_type == "last_1ep":
        epoch = 1
        global_train_examples_seen_this_epoch = 933120000
        train_batch_size = 1
    elif args.data_type == "next_1k":
        epoch = 1 if step < 432410 else 2
        global_train_examples_seen_this_epoch = train_state.get("global_train_examples_seen_this_epoch", train_state['global_train_examples_seen'])
        if step == 432410:
            global_train_examples_seen_this_epoch = 0 
    
    dataset = build_memmap_dataset(cfg, cfg.data)
    data_order_file_path=f"data/global_indices/global_indices_epoch{epoch}.npy"
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    print(f"\n Loaded dataset \n epoch: {epoch} \n global_train_examples_seen_this_epoch : {global_train_examples_seen_this_epoch}")
        
    instances = []
    batch_start = global_train_examples_seen_this_epoch
    for i in range(args.data_size):
        instances.append(global_indices[batch_start+i*train_batch_size])
        
    print(f"\n Loaded instances \n {instances[0]} - {instances[-1]}\n")
    subset_dataset = IndexedDataset(dataset, instances)
    dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False) 
    
    model = ExpOlmoForCausalLM.from_pretrained(f"checkpoints/pretrained/{step}/hf",
                                               attn_implementation="eager")    
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # import pdb; pdb.set_trace()
    batch_num = 0
    entropy_pred = 0     
    entropy_act = [0] * 32
    entropy_attention = [0] * 32
    act_sparsity = torch.zeros((32,11008), device=model.device)
    all_gold_probabilities = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Assuming batch is a dictionary with 'input_ids' and 'attention_mask'
            input_ids = batch['input_ids'].to(device)

            outputs = model(input_ids=input_ids, output_attentions=True)
            
            # entropy of prediction prob
            logits = outputs.logits
            bs, seq_len, _ = logits.shape
            batch_num += bs
            
            logits = logits[:, :-1, :]
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # (bs, seq_len-1)
            entropy_pred += torch.sum(entropy).item()/(logits.shape[1])
            
            # NTP probabilities
            logits = logits.contiguous() 
            shift_labels = input_ids[..., 1:].contiguous() 

            shift_probabilities = torch.softmax(logits, dim=-1)       
            gold_probabilities = torch.gather(shift_probabilities, -1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
            all_gold_probabilities.append(gold_probabilities)
                
            # entropy of Attention scores 
            mask = torch.tril(torch.ones(seq_len, seq_len, device=model.device))
            for layer_idx, attention in enumerate(outputs.attentions):
                log_probs = torch.log(attention + 1e-9) * mask #(bs, num_heads, seq_len, seq_len)
                entropy = -torch.sum(attention * log_probs, dim=-1) #(bs, num_heads, seq_len)
                entropy = torch.mean(entropy, dim=1) # (bs, seq_len)
                entropy_attention[layer_idx] += torch.sum(entropy).item()/(entropy.shape[-1])
                
            # entropy of MLP activations 
            for layer_idx, activation in enumerate(outputs.activations):
                probs = F.softmax(activation, dim=-1)
                log_probs = torch.log(probs + 1e-9)
                entropy = -torch.sum(probs * log_probs, dim=-1) # (bs, seq_len)
                entropy_act[layer_idx] += torch.sum(entropy).item()/(entropy.shape[-1])
                
                reshaped_activation = torch.abs(activation).view(-1, 11008)
                summed_activation = torch.sum(reshaped_activation, dim=0)
                summed_activation /= seq_len
                act_sparsity[layer_idx] += summed_activation
            # Clear memory
            del input_ids, outputs, logits, probs, log_probs, entropy, mask, shift_probabilities, gold_probabilities
            torch.cuda.empty_cache() 
        
    # import pdb; pdb.set_trace()
    entropy_pred /= batch_num
    entropy_act = [ act/batch_num for act in entropy_act]
    entropy_attention = [ att/batch_num for att in entropy_attention]
    
    # activation sparsity
    act_sparsity /= batch_num
    probabilities = act_sparsity / torch.sum(act_sparsity, dim=1, keepdim=True)
    probabilities = probabilities + 1e-9
    log_probabilities = torch.log(probabilities)
    entropy_act_sparsity = -torch.sum(probabilities * log_probabilities, dim=-1)
    entropy_act_sparsity = entropy_act_sparsity.tolist()
    
    # Calculate mean, median, and mode of NTP probabilties
    all_gold_probabilities = torch.cat(all_gold_probabilities, dim=0)
    mean_probability = all_gold_probabilities.mean().item()
    median_probability = all_gold_probabilities.median().item()
    
    # calculate mode 
    num_bins = 100
    hist = torch.histc(all_gold_probabilities, bins=100, min=0, max=1)
    max_bin = hist.argmax()
    # Calculate the bin edges (linearly spaced)
    bin_edges = torch.linspace(0, 1, steps=num_bins+1)
    mode_probabilitiy = bin_edges[max_bin].item() + 0.005

    to_save = {
        "entropy_pred": entropy_pred,
        "entropy_act": entropy_act,
        "entropy_attention":entropy_attention,
        "entropy_act_sparsity": entropy_act_sparsity,
        "pred_distribution": {
            "mean": mean_probability,
            "median": median_probability,
            "mode": mode_probabilitiy
        }
    }
    print(to_save)
    write_json_file(f"checkpoints/pretrained/{step}/entropy_{args.data_type}.json", to_save)
    all_gold_probabilities = all_gold_probabilities.float().half().detach()
    torch.save(all_gold_probabilities, f"checkpoints/pretrained/{step}/gold_probabilities_{args.data_type}.pt")
    
def write_json_file(file_path, res):
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Wrote json file to: {file_path}!")

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res

def report(data_type):
    import os
    filelist = [int(n) for n in os.listdir("checkpoints/pretrained") if "_" not in n]
    result = defaultdict(list)
    step_temp = "step|"
    entropy_pred_temp = "entropy_pred|"
    for step in sorted(filelist):
        eval_path = f"checkpoints/pretrained/{step}/entropy_{data_type}.json"
        if not os.path.isfile(eval_path):
            print(f"No file for {eval_path}.")
            continue
        data = read_json_file(eval_path)
        step_temp += f"{step}|"
        entropy_pred_temp += f"{data['entropy_pred']}|"
        for layer_idx in range(len(data['entropy_act'])):
            result[f"entropy_act_{layer_idx}"].append(str(data['entropy_act'][layer_idx]))
            result[f"entropy_attention_{layer_idx}"].append(str(data['entropy_attention'][layer_idx]))
            result[f"entropy_act_sparsity{layer_idx}"].append(str(data['entropy_act_sparsity'][layer_idx]))
    
    print(step_temp)
    print(entropy_pred_temp)
    result = dict(sorted(result.items()))
    for k,v in result.items():
        print(f"{k}|{'|'.join(v)}")

# def get_next_1k_instances(start_idx: int) -> list[list[int]]:
#     batch_start = start_idx * batch_size
#     batch_instances = []
#     for i in range(1000):
#         index = global_indices[batch_start+i*batch_size]
#         token_ids = dataset[index]["input_ids"].tolist()
#         batch_instances.append(token_ids)
#     return batch_instances

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--data_size", type=int, default=4)
    parser.add_argument("--data_type", type=str, default="last_1k")
    parser.add_argument("--report", type=bool, default=False)


    
    # parser.add_argument("--model-path", type=str, default=None)
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--data-path", type=str, default="./data/conflictqa")
    # parser.add_argument("--data-type", type=str, default="Q2C_gen")
    # parser.add_argument("--answers-file", type=str, default=None)
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--top_k", type=int, default=10)
    # parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument("--num_return_sequences", type=int, default=1)
    # parser.add_argument("--max_new_tokens", type=int, default=30)
    # parser.add_argument("--min_length", type=int, default=3)
    args = parser.parse_args()
    if args.step:
        main(args)
        
    if args.report:
        report(args.data_type)