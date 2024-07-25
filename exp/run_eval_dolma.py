from hf_olmo import OLMoForCausalLM
from transformers import AutoTokenizer
import argparse
import json
from exp.dataset import IndexedDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch 
from tqdm import tqdm
import os
from collections import defaultdict

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res
def write_json_file(file_path, res):
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Wrote json file to: {file_path}!")

def run_dolma_eval(args):
    
    revisions_dict = {}
    with open('checkpoints/pretrained_hf/revisions.txt', 'r') as file:
        for line in file:
            line = line.strip()
            step_str, rest = line.split('-', 1)
            step_num = int(step_str.replace('step', ''))
            revisions_dict[step_num] = line

    model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B", revision=revisions_dict[args.step], cache_dir=f"/data/jiyeon/OLMo/checkpoints/pretrained_hf/{args.step}")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
    print(f"\n Loaded model from {revisions_dict[args.step]}")
    if args.data_type == "next_1k_new": 
        data_path = f"data/dolma/step_{args.data_step}_next_1k.json"
    else:
        data_path = f"data/dolma/step_{args.data_step}_prev_1.json"
    dataset = read_json_file(data_path)
    print(f"\n Loaded dolma dataset \n length: {len(dataset)} \n from: {data_path}") #example: {dataset[0]}
    dataset = [d['text'] for d in dataset] 
    instances = [d for d in range(len(dataset))]
    subset_dataset = IndexedDataset(dataset, instances, tokenizer=tokenizer)        
    dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False) 
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_gold_probabilities = []
    all_mask = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Assuming batch is a dictionary with 'input_ids' and 'attention_mask'
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # entropy of prediction prob
            logits = outputs.logits
            bs, seq_len, _ = logits.shape
            
            logits = logits[:, :-1, :]
            
            
            
            # NTP probabilities
            logits = logits.contiguous() 
            shift_labels = input_ids[..., 1:].contiguous() 

            shift_probabilities = torch.softmax(logits, dim=-1)       
            gold_probabilities = torch.gather(shift_probabilities, -1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
            all_gold_probabilities.append(gold_probabilities)
            mask = (shift_labels != 1)
            all_mask.append(mask)
            
            del gold_probabilities, shift_probabilities, shift_labels, logits, input_ids, attention_mask
            torch.cuda.empty_cache()
            

    # Calculate mean, median, and mode of NTP probabilties
    all_gold_probabilities = torch.cat(all_gold_probabilities, dim=0)
    print("Before masking", all_gold_probabilities.shape)
    all_mask = torch.cat(all_mask, dim=0)
    all_length = torch.sum(all_mask, dim=1)
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

    position = 0
    average_prob = torch.zeros(len(dataset))
    for idx in range(len(dataset)):
        length = all_length[idx].item()
        prob_sum = all_gold_probabilities[position:position+length].sum()
        avg = prob_sum/length
        average_prob[idx] = avg
        position += length
        
    # import pdb; pdb.set_trace()
    average_prob_list = average_prob.tolist()
    instance_mean_probability = average_prob.mean().item()
    to_save = {
            "pred_distribution": {
                "mean": mean_probability,
                "median": median_probability,
                "mode": mode_probabilitiy,
                "variance": variance_probability,
                "instance_mean_probability": instance_mean_probability
            },
            "instance_prob_raw" : average_prob_list    
    }

    all_gold_probabilities = all_gold_probabilities.float().detach().cpu()
    save_path = f"/data/jiyeon/OLMo/checkpoints/pretrained_hf/{args.step}/eval_dolma"
    os.makedirs(save_path, exist_ok=True)
    write_json_file(f"{save_path}/dolma{args.data_step}_model{args.step}_{args.data_type}.json", to_save)
    torch.save(all_gold_probabilities, f"{save_path}/dolma{args.data_step}_model{args.step}_{args.data_type}_gold_prob.pt")
    
def convert_to_instance(full_mask, prob_tensor):
    reshaped_prob_tensor = torch.full((full_mask.shape), fill_value=0.0)
    position = 0
    for i in range(full_mask.shape[0]):
        num_valid_tokens = full_mask[i].sum().item()
        reshaped_prob_tensor[i, :num_valid_tokens] = prob_tensor[position:position + num_valid_tokens]
        position += num_valid_tokens
    return reshaped_prob_tensor

def attention_mask(data_step):
    data = read_json_file(f"data/dolma/step_{data_step}_next_1k.json")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
    full_mask = []
    for item in data:
        attention_mask = tokenizer(item['text'], max_length=2048, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"].squeeze(0) 
        full_mask.append(attention_mask[1:])
    full_mask = torch.stack(full_mask)
    
    return full_mask

def report(args):
    model_steps = [0, 1000, 2000, 3000, 4000, 5000, 6000, 110000, 194000, 278000, 432000, 557000]
    
    label_tensor = torch.load(f"data/dolma/token_classified/token_classification_dolma_{args.data_step}.pt")
    mask_NE = label_tensor[:, 1:] == 0 #named entity 
    mask_semantic = label_tensor[:, 1:] == 1 #semantic token 
    mask_syntactic = label_tensor[:, 1:] == 2 #syntactic token 
    full_mask = attention_mask(args.data_step)
    
    result = defaultdict(list)
    for model_step in model_steps:
        prob_tensor = torch.load(f"checkpoints/pretrained_hf/{model_step}/eval_dolma/dolma{args.data_step}_model{model_step}_next_1k_new_gold_prob.pt")
        
        reshaped_prob_tensor = convert_to_instance(full_mask, prob_tensor)
        
        mean_ne = reshaped_prob_tensor[mask_NE].mean().item()
        mean_semantic = reshaped_prob_tensor[mask_semantic].mean().item()
        mean_syntactic = reshaped_prob_tensor[mask_syntactic].mean().item()
        result['steps'].append(str(model_step))
        result['ne'].append(str(mean_ne))
        result['semantic'].append(str(mean_semantic))
        result['syntactic'].append(str(mean_syntactic))
    
    for k,v in result.items():
        print(f"{k}|{'|'.join(v)}")

def compare_delta(args):
    model_step, model_step_1k = args.step, args.step+1000
    
    label_tensor = torch.load(f"data/dolma/token_classified/token_classification_dolma_{model_step}.pt")
    mask_NE = label_tensor[:, 1:] == 0 #named entity 
    mask_semantic = label_tensor[:, 1:] == 1 #semantic token 
    mask_syntactic = label_tensor[:, 1:] == 2 #syntactic token 
    full_mask = attention_mask(model_step)
    
    result = defaultdict(list)
    prob_tensor = torch.load(f"checkpoints/pretrained_hf/{model_step}/eval_dolma/dolma{model_step}_model{model_step}_next_1k_new_gold_prob.pt")
    reshaped_prob_tensor = convert_to_instance(full_mask, prob_tensor)
    
    prob_tensor_1k = torch.load(f"checkpoints/pretrained_hf/{model_step_1k}/eval_dolma/dolma{model_step}_model{model_step_1k}_next_1k_new_gold_prob.pt")
    reshaped_prob_tensor_1k = convert_to_instance(full_mask, prob_tensor_1k)
    diff = reshaped_prob_tensor_1k - reshaped_prob_tensor
    
    
    print(f"NE after train|{reshaped_prob_tensor_1k[mask_NE].mean().item()}")
    print(f"semantic after train|{reshaped_prob_tensor_1k[mask_semantic].mean().item()}")
    print(f"syntactic after train|{reshaped_prob_tensor_1k[mask_syntactic].mean().item()}")
    
    print(f"NE difference train|{diff[mask_NE].mean().item()}")
    print(f"semantic difference train|{diff[mask_semantic].mean().item()}")
    print(f"syntactic difference train|{diff[mask_syntactic].mean().item()}")
    
    diff_log = -torch.log(reshaped_prob_tensor_1k + 1e-9) + -torch.log(reshaped_prob_tensor + 1e-9)
    
    print(f"NE difference of -log(p) train|{diff_log[mask_NE].mean().item()}")
    print(f"semantic difference of -log(p) train|{diff_log[mask_semantic].mean().item()}")
    print(f"syntactic difference of -log(p) train|{diff_log[mask_syntactic].mean().item()}")
    
    #     result['steps'].append(str(model_step))
    #     result['ne'].append(str(mean_ne))
    #     result['semantic'].append(str(mean_semantic))
    #     result['syntactic'].append(str(mean_syntactic))
    
    # for k,v in result.items():
    #     print(f"{k}|{'|'.join(v)}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--data_step", type=int, default=None)
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--report", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    
    if args.report:
        if args.data_step is not None:
            report(args)
        elif args.step is not None:
            compare_delta(args)
    else:
        run_dolma_eval(args)
        
    """
    python -m exp.run_eval_dolma --report True --data_step 0
    python -m exp.run_eval_dolma --report True --step 0
    """