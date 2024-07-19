import argparse
import torch
import json
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .modeling_olmo_hf import ExpOlmoForCausalLM
import numpy as np
import torch.nn.functional as F
from olmo_old.config import TrainConfig
from olmo_old.data import build_memmap_dataset
from torch.utils.data import Dataset, DataLoader
from .dataset import IndexedDataset
import os

def main(args):     
               
    dataloader, model, instances, model_path = load_model(args)
    if args.temperature:
        return get_temperature(dataloader, model, model_path)
    # import pdb; pdb.set_trace()
    batch_num = 0
    non_padding_count = 0 
    non_padding_count_att = 0 
    
    entropy_pred = 0     
    entropy_act = [0] * 32
    entropy_gate_act = [0] * 32
    entropy_attention = [0] * 32
    act_sparsity = torch.zeros((32,11008), device=model.device)
    all_gold_probabilities = []
    all_mask = []
    
    name = '_'+args.data_path if args.data_path else ''
    
    mlp_temperature, attn_temperature = None, 1.0 
    if args.mlp_temp_path is not None:
        mlp_temperature = torch.load(args.mlp_temp_path).to(model.device)
        temp_name = args.mlp_temp_path.split("/")[-1].split(".pt")[0]
        name += f"-mlp_temp_{temp_name}"
    if args.attn_initial_temp is not None:
        attn_temperature = args.attn_initial_temp
        name += f"-att_temp_{str(attn_temperature)}"
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Assuming batch is a dictionary with 'input_ids' and 'attention_mask'
            input_ids = batch['input_ids'].to(model.device)

            outputs = model(input_ids=input_ids, output_attentions=True, temperature=attn_temperature, mlp_temperature=mlp_temperature)
            
            # entropy of prediction prob
            logits = outputs.logits
            bs, seq_len, _ = logits.shape
            batch_num += bs
            
            logits = logits[:, :-1, :]
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # (bs, seq_len-1)
            
            # if 'attention_mask' in batch:
            if args.data_type == "cpt" and 'attention_mask' in batch:
                attention_mask = batch['attention_mask'].to(model.device)
                attention_mask = attention_mask[:, :-1] # :-1 => 1: #(bs, seq_len-1)
                # Apply mask to entropy - set entropy of padding positions to zero
                masked_entropy = entropy * attention_mask

                # Sum the entropy over all tokens (ignoring padding) and normalize by the number of non-padding tokens
                non_padding_count += torch.sum(attention_mask).item()
                non_padding_count_att += torch.sum(batch['attention_mask']).item()
                
                entropy_pred += torch.sum(masked_entropy).item()
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
            # import pdb; pdb.set_trace()
            # entropy of Attention scores 
            mask = torch.tril(torch.ones(seq_len, seq_len, device=model.device))
            for layer_idx, attention in enumerate(outputs.attentions):
                log_probs = torch.log(attention + 1e-4) * mask #(bs, num_heads, seq_len, seq_len)
                entropy = -torch.sum(attention * log_probs, dim=-1) #(bs, num_heads, seq_len)
                entropy = torch.mean(entropy, dim=1) # (bs, seq_len)
                if args.data_type == "cpt" and 'attention_mask' in batch:
                    tempmask = entropy[batch['attention_mask'] == 1]
                    entropy_attention[layer_idx] += tempmask.sum().item()
                else:
                    entropy_attention[layer_idx] += torch.sum(entropy).item()/(entropy.shape[-1])

            # import pdb; pdb.set_trace()
            # entropy of MLP activations             
            for layer_idx, activation in enumerate(outputs.activations):
                probs = F.softmax(torch.abs(activation), dim=-1)
                log_probs = torch.log(probs + 1e-9)
                entropy = -torch.sum(probs * log_probs, dim=-1) # (bs, seq_len)
                if args.data_type == "cpt" and 'attention_mask' in batch:
                    tempmask = entropy[batch['attention_mask'] == 1]
                    entropy_act[layer_idx] += tempmask.sum().item()
                else:
                    entropy_act[layer_idx] += torch.sum(entropy).item()/(entropy.shape[-1])
                
                reshaped_activation = torch.abs(activation).view(-1, 11008)
                if args.data_type == "cpt" and 'attention_mask' in batch:
                    flat_attention_mask = batch['attention_mask'].view(-1)
                    reshaped_activation = reshaped_activation[flat_attention_mask == 1]
                    summed_activation = torch.sum(reshaped_activation, dim=0)
                    act_sparsity[layer_idx] += summed_activation
                else:    
                    summed_activation = torch.sum(reshaped_activation, dim=0)
                    summed_activation /= activation.shape[1]
                    act_sparsity[layer_idx] += summed_activation
            
            # for layer_idx, activation in enumerate(outputs.gate_activations):
            #     probs = F.softmax(torch.abs(activation), dim=-1)
            #     log_probs = torch.log(probs + 1e-9)
            #     entropy = -torch.sum(probs * log_probs, dim=-1) # (bs, seq_len)
            #     entropy_gate_act[layer_idx] += torch.sum(entropy).item()/(entropy.shape[-1])
                
            # Clear memory
            del input_ids, outputs, logits, probs, log_probs, entropy, mask, shift_probabilities, gold_probabilities
            torch.cuda.empty_cache() 
        
    # import pdb; pdb.set_trace()
    if non_padding_count == 0:
        entropy_pred /= batch_num 
        entropy_act = [ act/batch_num for act in entropy_act]
        entropy_attention = [ att/batch_num for att in entropy_attention]
        entropy_gate_act = [ act/batch_num for act in entropy_gate_act]
        act_sparsity /= batch_num
    else:
        entropy_pred /= non_padding_count
        entropy_act = [ act/non_padding_count_att for act in entropy_act]
        entropy_attention = [ att/non_padding_count_att for att in entropy_attention]
        entropy_gate_act = [ act/non_padding_count_att for act in entropy_gate_act]
        act_sparsity /= non_padding_count_att
        

    to_save = {
        "entropy_pred": entropy_pred,
        "entropy_act": entropy_act,
        "entropy_gate_act":entropy_gate_act,
        "entropy_attention":entropy_attention,
        "instances": [int(d) for d in instances]
    }
    
    
    # activation sparsity
        # probabilities = act_sparsity / torch.sum(act_sparsity, dim=1, keepdim=True)
    probabilities = torch.softmax(act_sparsity, dim=-1)
    log_probabilities = torch.log(probabilities + 1e-9)
    entropy_act_sparsity = -torch.sum(probabilities * log_probabilities, dim=-1)
    entropy_act_sparsity = entropy_act_sparsity.tolist()    
    to_save.update({
        "entropy_act_sparsity": entropy_act_sparsity,
    })
    probabilities = probabilities.float().detach().cpu()
    
    
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

    to_save.update({
        "pred_distribution": {
            "mean": mean_probability,
            "median": median_probability,
            "mode": mode_probabilitiy,
            "variance": variance_probability,
        },
    })
    print(to_save)
    
    # Save result
    all_gold_probabilities = all_gold_probabilities.float().half().detach()
    if args.data_step is None :
        torch.save(act_sparsity, f"{model_path}/mlp_activation_sparsity_raw_{args.data_type}{name}.pt")
        write_json_file(f"{model_path}/entropy_{args.data_type}{name}.json", to_save)
        torch.save(all_gold_probabilities, f"{model_path}/gold_probabilities_{args.data_type}{name}.pt")
    else:
        path = 'checkpoints/pretrained/dolma_prob' 
        os.makedirs(f"{path}/dolma{args.data_step}", exist_ok=True)
        write_json_file(f"{path}/dolma{args.data_step}/model{args.step}_entropy_dolma{args.data_step}_new.json", to_save)
        torch.save(all_gold_probabilities, f"{path}/dolma{args.data_step}/model{args.step}_gold_prob_dolma{args.data_step}_new.pt")
        torch.save(act_sparsity, f"{path}/dolma{args.data_step}/model{args.step}_mlp_act_dolma{args.data_step}_new.pt")
    
def load_model(args):
            
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
    elif args.data_type in ["cpt", "next_1k_new", "prev_1"] :
        pass
    else:
        raise ValueError("Invalud option chosen for data_type")
    
    if args.data_type == "cpt":
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
        dataset = read_json_file(f"data/corpus/raw/{args.data_path}.json")
        print(f"\n Loaded CPT dataset from {args.data_path} \n length: {len(dataset)} \n example: {dataset[0]}")
        dataset = [d['text'] for d in dataset]
        instances = [d for d in range(len(dataset))]
        subset_dataset = IndexedDataset(dataset, instances, tokenizer=tokenizer, seq_len=1024)
    elif args.data_type in ["next_1k_new","prev_1"]:
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
        if args.data_type == "next_1k_new": 
            dataset = read_json_file(f"data/dolma/step_{args.step}_next_1k.json")
        else:
            dataset = read_json_file(f"data/dolma/step_{args.data_step}_prev_1.json")
            
        print(f"\n Loaded dolma dataset \n length: {len(dataset)} \n example: {dataset[0]}")
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
    if args.bf16: 
        model.to(dtype=torch.bfloat16)
          
    return dataloader, model, instances, model_path

def get_temperature(dataloader, model, model_path):
    batch_num = 0
    temperature_candidates = [1.0 + i / 10 for i in range(10)]
    entropy_attention = defaultdict(lambda: defaultdict(float))
    import pdb; pdb.set_trace()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(model.device)

            outputs = model(input_ids=input_ids, output_attentions=False)
            bs, seq_len, _ = outputs.logits.shape
            batch_num += bs
            
            # Collect attention weights for all layers at once
            all_attn_weights_raw = [attn_weights.detach() for attn_weights in outputs.all_self_attn_weights_raw]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=torch.device("cuda")))
            all_attn_weights_raw = torch.cat(all_attn_weights_raw, dim=0).to(model.device) 
            
            del input_ids, outputs
            torch.cuda.empty_cache()
            
            for temperature in temperature_candidates:

                # Apply temperature scaling and softmax
                attn_weights = all_attn_weights_raw / temperature #(32, bs, num_heads, seq_len, seq_len)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32) 

                # Compute log probabilities and entropy
                log_probs = torch.log(attn_weights + 1e-9) * mask
                entropy = -torch.sum(attn_weights * log_probs, dim=-1)  # (32, bs, num_heads, seq_len)
                entropy = entropy.mean(dim=[2, 3])  # (32, bs)

                for layer_idx in range(32):
                    entropy_attention[temperature][layer_idx] +=  entropy[layer_idx].sum().item()

                del attn_weights, log_probs, entropy
                torch.cuda.empty_cache()
            
            # for layer_idx, attn_weights_raw in enumerate(outputs.all_self_attn_weights_raw):
            #     attn_weights_raw = attn_weights_raw.detach() #(bs, 32, seq_len, seq_len)
                
            #     mask = torch.tril(torch.ones(seq_len, seq_len, device=torch.device("cuda")))
            #     for temperature in temperature_candidates:
            #         attn_weights = attn_weights_raw / temperature  # Apply temperature scaling
            #         attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # Softmax
            #         log_probs = torch.log(attn_weights + 1e-9) * mask  # Compute log probabilities with mask #(bs, num_heads, seq_len, seq_len)
            #         entropy = -torch.sum(attn_weights * log_probs, dim=-1)  # Compute entropy #(bs, num_heads, seq_len)
            #         layer_entropy = entropy.mean(dim=[1, 2])  # Average over heads and sequence length # (bs)
            #         # entropy_item = layer_entropy.mean().item()  # Average over batch
            #         # entropy_attention[temperature][layer_idx].append(entropy_item)  # Store entropy
            #         entropy_attention[temperature][layer_idx] += torch.sum(entropy).item()
                    
            #         del attn_weights, log_probs, entropy, layer_entropy
            #         torch.cuda.empty_cache()

                # del attn_weights_raw
                # torch.cuda.empty_cache()

    import pdb; pdb.set_trace()
    avg_entropy_attention = defaultdict(dict)
    for temperature in temperature_candidates:
        print("-"*50)
        for layer_idx in entropy_attention[temperature]:
            avg_entropy = entropy_attention[temperature][layer_idx] / batch_num
            avg_entropy_attention[temperature][layer_idx] = avg_entropy
            print(f"{temperature} | {layer_idx} | {avg_entropy}")

        overall_avg_entropy = sum(avg_entropy_attention[temperature].values()) / len(avg_entropy_attention[temperature])
        avg_entropy_attention[temperature]['average'] = overall_avg_entropy
        print(f"{temperature} | overall average | {overall_avg_entropy}")

    write_json_file(f"{model_path}/entropy_attention_temperature.json", avg_entropy_attention)

def get_temperature_to_Delete(dataloader, model):
    
    batch_num = 0
    entropy_attention = defaultdict(dict)
    temperature_candidates = [1.0+i/10 for i in range(20)]
    # all_attention_weights = defaultdict(list)
    all_attention_weights = torch.zeros((args.data_size, 32, 32, 2048, 2048), device=torch.device('cpu'))
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # import pdb; pdb.set_trace()
            input_ids = batch['input_ids'].to(model.device)

            outputs = model(input_ids=input_ids, output_attentions=False)
            
            logits = outputs.logits
            bs, seq_len, _ = logits.shape
            batch_num += bs
            
            for layer_idx, attn_weights_raw in enumerate(outputs.all_self_attn_weights_raw):
                # all_attention_weights[layer_idx].append(attn_weights_raw.detach().cpu())
                all_attention_weights[batch_num:batch_num+bs, layer_idx] = attn_weights_raw.detach().cpu()
            # import pdb; pdb.set_trace()
            del input_ids, outputs, logits, attn_weights_raw
            torch.cuda.empty_cache() 
    
    del model, dataloader
    torch.cuda.empty_cache() 
    import pdb; pdb.set_trace()
        
    mask = torch.tril(torch.ones(seq_len, seq_len, device=torch.device("cuda")))
    all_attention_weights = all_attention_weights.cpu()
    for temperature in temperature_candidates:
        print("-"*50)
        attn_weights = all_attention_weights / temperature  #(bs, num_layers, num_heads, seq_len, seq_len) 
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32) #(bs, num_layers, num_heads, seq_len, seq_len)        
        log_probs = torch.log(attn_weights + 1e-9) * mask #(bs, num_layers, num_heads, seq_len, seq_len)
        entropy = -torch.sum(attn_weights * log_probs, dim=-1) #(bs, num_layers, num_heads, seq_len)
        
        for layer_idx in range(attn_weights.shape[1]):
            layer_entropy = entropy[:, layer_idx, :, :].mean(dim=[1, 2])  # (bs)
            entropy_item = layer_entropy.mean().item()
            entropy_attention[temperature][layer_idx] = entropy_item
            print(f"{temperature} | {layer_idx} | {entropy_item}")
            
            del layer_entropy
            torch.cuda.empty_cache()
        
        avg_entropy = sum([ent for ent in entropy_attention[temperature].values()]) / len(entropy_attention[temperature])
        entropy_attention[temperature]["average"] = avg_entropy
        print(f"{temperature} | average | {avg_entropy}")
            
    # 10-15, 23,26
    # import pdb; pdb.set_trace()
    # all_attention_weights_gpu = {}
    # for layer_idx, layer_attention_weights in all_attention_weights.items():
    #     if not (10 <= layer_idx <= 15 or 23 <= layer_idx <= 26):
    #         layer_attention_weights = torch.cat(layer_attention_weights, dim=0).to(torch.device("cuda"))
    #         all_attention_weights_gpu[layer_idx]=layer_attention_weights
    #         print(f"{layer_idx}, {layer_attention_weights.shape}")
    # for temperature in temperature_candidates:
    #     attn_weights = all_attention_weights / temperature  #(bs, num_layers, num_heads, seq_len, seq_len) 
    #     attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32) #(bs, num_layers, num_heads, seq_len, seq_len)        
    #     log_probs = torch.log(attn_weights + 1e-9) * mask #(bs, num_layers, num_heads, seq_len, seq_len)
    #     entropy = -torch.sum(attn_weights * log_probs, dim=-1) #(bs, num_layers, num_heads, seq_len)        
    #     entropy = entropy.transpose(0,1) # (num_layers, bs, num_heads, seq_len)
    #     entropy = torch.mean(entropy, dim=2) # (num_layers, bs, seq_len)
    #     entropy = torch.mean(entropy, dim=2) # (num_layers, bs)
    #     entropy_attention[layer_idx] += torch.sum(entropy).item()/(entropy.shape[-1])
        
    #     for layer_idx, layer_attention_weights in all_attention_weights_gpu.items():
    #         attn_weights = layer_attention_weights / temperature
    #         attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    #         log_probs = torch.log(attn_weights + 1e-4) * mask #(bs, num_heads, seq_len, seq_len)
    #         entropy = -torch.sum(attn_weights * log_probs, dim=-1) #(bs, num_heads, seq_len)
    #         entropy = torch.mean(entropy, dim=1) # (bs, seq_len)
    #         entropy_item = torch.sum(entropy).item()/(entropy.shape[-1])
    #         entropy_attention[temperature].append(entropy_item) 
    #         print(f"{temperature} | {layer_idx} |{entropy_item}")     
            
    #         del attn_weights, log_probs, entropy
    #         torch.cuda.empty_cache() 
    #     print(f"{temperature} | average | {sum(entropy_attention[temperature])/len(entropy_attention[temperature])}")

def write_json_file(file_path, res):
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Wrote json file to: {file_path}!")

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res

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

def save_dolma(loc, step=None):
            
    if loc == "prev_1k":
        model_path = f"checkpoints/pretrained/{step}"
        train_state = torch.load(f"{model_path}/train.pt")
        
        if step <= 432410:
            epoch = 1
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen']
        else:
            epoch = 2
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen_this_epoch']
            
        global_train_examples_seen_this_epoch -= 2160000
    elif loc == "first_1k":
        epoch = 1
        global_train_examples_seen_this_epoch = 0
    elif loc == "next_1k":
        model_path = f"checkpoints/pretrained/{step}"
        train_state = torch.load(f"{model_path}/train.pt")
        
        if step < 432410:
            epoch = 1
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen']
        elif step == 432410:
            epoch = 2
            global_train_examples_seen_this_epoch = 0
        else:
            epoch = 2
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen_this_epoch']
        print(global_train_examples_seen_this_epoch)
    elif loc == "prev_1":
        model_path = f"checkpoints/pretrained/{step}"
        train_state = torch.load(f"{model_path}/train.pt")
        
        if step < 432410:
            epoch = 1
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen']
        elif step == 432410:
            epoch = 1
            global_train_examples_seen_this_epoch = 934005600
        else:
            epoch = 2
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen_this_epoch']
            
        global_train_examples_seen_this_epoch -= 2160
        
        
    data_order_file_path=f"data/global_indices/global_indices_epoch{epoch}.npy"
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    print(f"\n Loaded dataset \n epoch: {epoch} \n global_train_examples_seen_this_epoch : {global_train_examples_seen_this_epoch}")
    
    instances = []
    batch_start = global_train_examples_seen_this_epoch
    term = 2160 if loc != "prev_1" else 1
    for i in range(1000):
        instances.append(global_indices[batch_start+i*term])
        
    train_config_path = "configs/official/OLMo-7B_2160.yaml"    
    cfg = TrainConfig.load(train_config_path)
    dataset = build_memmap_dataset(cfg, cfg.data)
    
    # import pdb; pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
    to_save = []
    print("\nStart decoding dataset")
    for inst in tqdm(instances,total=1000):
        input_ids = dataset[inst]['input_ids']
        text = tokenizer.batch_decode([input_ids])
        to_save.append({
            "id": int(inst),
            "text": text[0]
        })
        
    write_json_file(f"data/dolma/{'prev_1/' if loc == 'prev_1' else ''}step_{step}_{loc}.json", to_save)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--data_step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--data_size", type=int, default=4)
    parser.add_argument("--data_type", type=str, default="last_1k")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_manual_start_num", type=int, default=None)
    parser.add_argument("--data_manual_epoch", type=int, default=None)
    parser.add_argument("--finetuned_path", type=str, default=None)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--temperature", type=bool, default=False)
    parser.add_argument("--save_dolma", type=bool, default=False)
    parser.add_argument("--attn_initial_temp", type=float, default=None)
    parser.add_argument("--mlp_temp_path", type=str, default=None)

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
    if args.save_dolma:
        save_dolma(args.data_type, args.step)
    else:
        if args.step or args.finetuned_path:
            main(args)
        