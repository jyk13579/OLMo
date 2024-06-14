import json
from collections import defaultdict 
import argparse
import glob
import torch

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res

EVAL_1 = ['eval_slotPubmed_loss', 'eval_original_slot_generation_original_gen_accuracy']
EVAL_2 = ['eval_slotParaphrase_loss', 'eval_original_slot_generation_paraphrase_gen_accuracy']
EVAL_3 = ['diff_casual_cf', 'eval_counterfactual_loss', 'eval_casual_loss']
EVAL_4_LOSS = ['eval_original_loss', 
                'eval_original_pubmed_orig_mean_probability',
                'eval_original_pubmed_orig_median_probability',
                'eval_original_pubmed_orig_variance_probability']
EVAL_4_PROB = ['eval_new_prob_diff_all', 
                'eval_new_prob_diff_0.0-0.1', 
                'eval_new_prob_diff_0.1-0.25', 
                'eval_new_prob_diff_0.25-0.5', 
                'eval_new_prob_diff_0.5-0.75', 
                'eval_new_prob_diff_0.75-0.9', 
                'eval_new_prob_diff_0.9-1',
                'eval_new_prob_count_all', 
                'eval_new_prob_count_0.0-0.1', 
                'eval_new_prob_count_0.1-0.25', 
                'eval_new_prob_count_0.25-0.5', 
                'eval_new_prob_count_0.5-0.75', 
                'eval_new_prob_count_0.75-0.9', 
                'eval_new_prob_count_0.9-1']
EVAL_4 = EVAL_4_LOSS + EVAL_4_PROB
EVAL_5_LOSS = ['eval_dolma_prev_loss', 
                'eval_original_dolma_prev1k_mean_probability',
                'eval_original_dolma_prev1k_median_probability',
                'eval_original_dolma_prev1k_variance_probability']
EVAL_5_PROB = ['eval_prev_prob_diff_all', 
                'eval_prev_prob_diff_0.0-0.1', 
                'eval_prev_prob_diff_0.1-0.25', 
                'eval_prev_prob_diff_0.25-0.5', 
                'eval_prev_prob_diff_0.5-0.75', 
                'eval_prev_prob_diff_0.75-0.9', 
                'eval_prev_prob_diff_0.9-1',
                'eval_prev_prob_count_all', 
                'eval_prev_prob_count_0.0-0.1', 
                'eval_prev_prob_count_0.1-0.25', 
                'eval_prev_prob_count_0.25-0.5', 
                'eval_prev_prob_count_0.5-0.75', 
                'eval_prev_prob_count_0.75-0.9', 
                'eval_prev_prob_count_0.9-1']
EVAL_5 = EVAL_5_LOSS + EVAL_5_PROB

def main(args):
    result = defaultdict(dict)
    path = args.path
    trainer_state = read_json_file(f"{path}/trainer_state.json")
    for log in trainer_state['log_history']:
        epoch = log.pop('epoch')
        step = log.pop('step')
        if "grad_norm" in log.keys() or "train_loss" in log.keys():
            pass
        else:
            for key, value in log.items():
                result[key][round(epoch)] = str(value)
                
    # calculate difference between casual & counterfactual
    for k in result['eval_casual_loss'].keys():
        dif = float(result['eval_casual_loss'][k]) - float(result['eval_counterfactual_loss'][k])
        result['diff_casual_cf'][k] = str(dif)
            
    # calculate probability difference categorized
    result = analyse_prob(path, result)
    
    keys = EVAL_1 + EVAL_2 + EVAL_3 + EVAL_4 + EVAL_5
    
    for key in keys:
        print_values(result[key], key)
    
def print_values(res, key):
    print(f"{key}|", "|".join(res.values()))
    
def print_prob(path): 
    result = analyse_prob(path)
    keys = EVAL_4_PROB + EVAL_5_PROB
    for key in keys:
        print_values(result[key], key)
        
def analyse_prob(path, result=defaultdict(dict)): 
        
    sorted_loaded_files = load_tensors(path, "pubmed_orig")
    for ep in range(len(sorted_loaded_files)):
        stats, counts = analyse(sorted_loaded_files[0]) if ep == 0 else analyse(sorted_loaded_files[0], sorted_loaded_files[ep])
        for k,v in stats.items():
            result[f"eval_new_prob_diff_{k}"][ep] = str(v)
            result[f"eval_new_prob_count_{k}"][ep] = str(counts[k])
        
    sorted_loaded_files = load_tensors(path, "dolma_prev1k")
    for ep in range(len(sorted_loaded_files)):
        stats, counts = analyse(sorted_loaded_files[0]) if ep == 0 else analyse(sorted_loaded_files[0], sorted_loaded_files[ep])
        for k,v in stats.items():
            result[f"eval_prev_prob_diff_{k}"][ep] = str(v)
            result[f"eval_prev_prob_count_{k}"][ep] = str(counts[k])
    
    return result
    
def load_tensors(folder_path, name):
    file_pattern = folder_path + f"/step_*_gold_probabilities_{name}.pt"
    files = glob.glob(file_pattern)
    
    loaded_files = {}
    for file in files:
        step_number = file.split("/")[-1].split('_')[1]  
        data = torch.load(file)  
        loaded_files[int(step_number)] = data  
        
    sorted_loaded_files = dict(sorted(loaded_files.items()))
    for idx, step in enumerate(sorted_loaded_files):
        sorted_loaded_files[idx] = sorted_loaded_files.pop(step)
        
    return sorted_loaded_files
    
def analyse(prob_pre, prob_ft=None):
    ranges = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1)]
    stats = {}
    counts = {}
    if prob_ft is not None:
        diff = prob_ft - prob_pre
    else:
        diff = prob_pre
    stats['all'] = diff.mean().item()
    counts['all'] = diff.shape[0]
    for r_min, r_max in ranges:
        mask = (prob_pre >= r_min) & (prob_pre < r_max)        
        filtered_diff = diff[mask]
        if filtered_diff.numel() > 0:
            mean_diff = filtered_diff.mean().item()
            # std_diff = filtered_diff.std().item()
            # min_diff = filtered_diff.min().item()
            # max_diff = filtered_diff.max().item()
        else:
            mean_diff = std_diff = min_diff = max_diff = None
        stats[f"{r_min}-{r_max}"] = mean_diff
        
        if prob_ft is not None:
            mask_ft = (prob_ft >= r_min) & (prob_ft < r_max)   
            count = prob_ft[mask_ft].shape[0]
        else:
            count = prob_pre[mask].shape[0]
        counts[f"{r_min}-{r_max}"] = count
        # .append({
        #     'range': ,
        #     'mean': mean_diff,
        #     'std': std_diff,
        #     'min': min_diff,
        #     'max': max_diff,
        #     'count': filtered_diff.shape[0]
        # })
    # for d in stats:
    #     print(d)
    return stats , counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    main(args)