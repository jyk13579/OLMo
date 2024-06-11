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
from dataset import IndexedDataset

def main(args):        
    step = args.step
    if not args.finetuned_path: 
        model_path = f"checkpoints/pretrained/{step}"
        train_state = torch.load(f"{model_path}/train.pt")
    else:
        model_path = args.finetuned_path
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
    elif args.data_type == "prob_dif":
        epoch = 1
        global_train_examples_seen_this_epoch = 933120000
        train_batch_size = 1
    elif args.data_type == "manual":
        global_train_examples_seen_this_epoch = args.data_manual_start_num
        epoch = args.data_manual_epoch
    elif args.data_type == "cpt":
        pass
    else:
        raise ValueError("Invalud option chosen for data_type")
    
    if args.data_type == "cpt":
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
        dataset = read_json_file(f"data/corpus/{args.data_path}.json")
        print(f"\n Loaded CPT dataset from {args.data_path} \n length: {len(dataset)} \n example: {dataset[0]}")
        dataset = [d['text'] for d in dataset]
        instances = [d for d in range(len(dataset))]
        subset_dataset = IndexedDataset(dataset, instances, tokenizer=tokenizer)
    else:
            
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
    
    model = ExpOlmoForCausalLM.from_pretrained(f"{model_path}{'/hf' if 'pretrained' in model_path else ''}", attn_implementation="eager")           
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
    all_mask = []

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
            import pdb; pdb.set_trace()
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask'].to(device)
                attention_mask = attention_mask[:, :-1]
                # Apply mask to entropy - set entropy of padding positions to zero
                masked_entropy = entropy * attention_mask

                # Sum the entropy over all tokens (ignoring padding) and normalize by the number of non-padding tokens
                entropy_sum = torch.sum(masked_entropy)
                non_padding_count = torch.sum(attention_mask)
                
                entropy_pred += entropy_sum.item() / non_padding_count.item()
            else:               
                entropy_pred += torch.sum(entropy).item()/(logits.shape[1])
        
            # NTP probabilities
            logits = logits.contiguous() 
            shift_labels = input_ids[..., 1:].contiguous() 

            shift_probabilities = torch.softmax(logits, dim=-1)       
            gold_probabilities = torch.gather(shift_probabilities, -1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
            all_gold_probabilities.append(gold_probabilities)
            mask = (shift_labels != 1)
            all_mask.append(mask)
                
            # expanded_attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (bs, seq_len, seq_len)
            # mask = mask * expanded_attention_mask  # Apply padding mask

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
    print("Before masking", all_gold_probabilities.shape)
    all_mask = torch.cat(all_mask, dim=0)
    all_gold_probabilities = all_gold_probabilities[all_mask]
    print("After masking", all_gold_probabilities.shape)
    mean_probability = all_gold_probabilities.mean().item()
    median_probability = all_gold_probabilities.median().item()
    variance_probability = torch.var(all_gold_probabilities, unbiased=False).item()

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
            "mode": mode_probabilitiy,
            "variance": variance_probability,
        },
        "instances": [int(d) for d in instances]
    }
    print(to_save)
    write_json_file(f"{model_path}/entropy_{args.data_type}{args.data_path if args.data_path else ''}.json", to_save)
    all_gold_probabilities = all_gold_probabilities.float().half().detach()
    torch.save(all_gold_probabilities, f"{model_path}/gold_probabilities_{args.data_type}{args.data_path if args.data_path else ''}.pt")
    
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
    result_prob = defaultdict(list)
    step_temp_prob = "step|"
    for step in sorted(filelist):
        eval_path = f"checkpoints/pretrained/{step}/entropy_{data_type}.json"
        if not os.path.isfile(eval_path):
            print(f"No file for {eval_path}.")
            continue
        data = read_json_file(eval_path)
        step_temp += f"{step}|"
        for k in data.keys():
            if isinstance(data[k], list):
                for layer_idx in range(len(data['entropy_act'])):
                    result[f"{k}_{layer_idx}"].append(str(data[k][layer_idx]))
            elif isinstance(data[k], dict):
                for k_temp in data[k].keys():
                    result[f"{k}_{k_temp}"].append(str(data[k][k_temp]))
            else:
                result[k].append(str(data[k]))      
        
        # gold prob
        prob_path = f"checkpoints/pretrained/{step}/gold_probabilities_{data_type}.pt"
        if not os.path.isfile(prob_path):
            print(f"No file for {prob_path}.")
            continue
        all_gold_probabilities = torch.load(prob_path).cpu()
        all_gold_probabilities = all_gold_probabilities.float()
        num_bins = 100
        hist = torch.histc(all_gold_probabilities, bins=num_bins, min=0, max=1)
        step_temp_prob += f"{step}|"
        for idx, element in enumerate(hist):
            result_prob[f"prob_{idx/100}"].append(str(int(element.item())))
            
        total_variance = torch.var(all_gold_probabilities, unbiased=False)
        result["pred_distribution_variance"].append(str(total_variance.item()))

    print(step_temp)
    result = dict(sorted(result.items()))
    for k,v in result.items():
        print(f"{k}|{'|'.join(v)}")

    print("\n\n\n", step_temp_prob)
    result_prob = dict(sorted(result_prob.items()))
    for k,v in result_prob.items():
        print(f"{k}|{'|'.join(v)}")
# def get_next_1k_instances(start_idx: int) -> list[list[int]]:
#     batch_start = start_idx * batch_size
#     batch_instances = []
#     for i in range(1000):
#         index = global_indices[batch_start+i*batch_size]
#         token_ids = dataset[index]["input_ids"].tolist()
#         batch_instances.append(token_ids)
#     return batch_instances

def analyse(prob_pre, prob_ft):
    ranges = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1)]
    stats = []    
    for r_min, r_max in ranges:
        mask = (prob_pre >= r_min) & (prob_pre < r_max)
        diff = prob_ft - prob_pre
        filtered_diff = diff[mask]
        if filtered_diff.numel() > 0:
            mean_diff = filtered_diff.mean().item()
            std_diff = filtered_diff.std().item()
            min_diff = filtered_diff.min().item()
            max_diff = filtered_diff.max().item()
        else:
            mean_diff = std_diff = min_diff = max_diff = None
        stats.append({
            'range': f"{r_min}-{r_max}",
            'mean': mean_diff,
            'std': std_diff,
            'min': min_diff,
            'max': max_diff,
            'count': filtered_diff.shape[0]
        })
    for d in stats:
        print(d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--data_size", type=int, default=4)
    parser.add_argument("--data_type", type=str, default="last_1k")
    parser.add_argument("--report", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_manual_start_num", type=int, default=None)
    parser.add_argument("--data_manual_epoch", type=int, default=None)
    parser.add_argument("--finetuned_path", type=str, default=None)


    
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
    if args.step or args.finetuned_path:
        main(args)
        
    if args.report:
        report(args.data_type)