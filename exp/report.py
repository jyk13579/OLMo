import json
from collections import defaultdict 
import argparse
import glob
import torch

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

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

# custom_ranges = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1)]
# custom_ranges = [(i/10, (i+1)/10) for i in range(10)]
# custom_ranges = [(i/2, (i+1)/2) for i in range(2)]
split_point = [0, 0.0025, 0.0067, 0.0183, 0.0498, 0.1353, 0.3679, 0.5, 1.01]
custom_ranges = [(split_point[i], split_point[i+1]) for i in range(len(split_point)-1)]
ranges = ['all'] + [f"{r_min}-{r_max}" for r_min, r_max in custom_ranges]
EVAL_4_PROB = [f"eval_new_prob_diff_{k}" for k in ranges] + [f"eval_new_prob_count_{k}" for k in ranges]

EVAL_4 = EVAL_4_LOSS + EVAL_4_PROB
EVAL_5_LOSS = ['eval_dolma_prev_loss', 
                'eval_original_dolma_prev1k_mean_probability',
                'eval_original_dolma_prev1k_median_probability',
                'eval_original_dolma_prev1k_variance_probability']

EVAL_5_PROB = [f"eval_prev_prob_diff_{k}" for k in ranges] + [f"eval_prev_prob_count_{k}" for k in ranges]
                
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
    
    # keys = EVAL_1 + EVAL_2 + EVAL_3 + EVAL_4 + EVAL_5
    keys = EVAL_4_PROB + EVAL_5_PROB
    
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
        stats, counts, _ = analyse(sorted_loaded_files[0]) if ep == 0 else analyse(sorted_loaded_files[0], sorted_loaded_files[ep])
        # import pdb; pdb.set_trace()
        for k,v in stats.items():
            result[f"eval_new_prob_diff_{k}"][ep] = str(v)
            result[f"eval_new_prob_count_{k}"][ep] = str(counts[k])
        
    sorted_loaded_files = load_tensors(path, "dolma_prev1k")
    for ep in range(len(sorted_loaded_files)):
        stats, counts, _ = analyse(sorted_loaded_files[0]) if ep == 0 else analyse(sorted_loaded_files[0], sorted_loaded_files[ep])
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
    # import pdb; pdb.set_trace()
        
    sorted_loaded_files = dict(sorted(loaded_files.items()))
    loaded_files = {}
    for idx, step in enumerate(sorted_loaded_files.keys()):
        loaded_files[idx] = sorted_loaded_files[step]
        
    return loaded_files
    
def analyse(prob_pre, prob_ft=None, ranges=custom_ranges):
    stats = {}
    counts = {}
    result = {}
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
            mean_orig = prob_pre[mask].mean().item()
            mean_diff = filtered_diff.mean().item()
            std_diff = filtered_diff.std().item()
            min_diff = filtered_diff.min().item()
            max_diff = filtered_diff.max().item()
        else:
            mean_diff = std_diff = min_diff = max_diff = None
        stats[f"{r_min}-{r_max}"] = mean_diff
        
        if prob_ft is not None:
            mask_ft = (prob_ft >= r_min) & (prob_ft < r_max)   
            count = prob_ft[mask_ft].shape[0]
        else:
            count = prob_pre[mask].shape[0]            
        counts[f"{r_min}-{r_max}"] = count
        
        result[f"{r_min}-{r_max}"] = {
                'orig_mean': mean_orig,
                'orig_count': filtered_diff.shape[0],
                'mean': mean_diff,
                'std': std_diff,
                'min': min_diff,
                'max': max_diff,
                'count': count
            }
    # if prob_ft is not None:
    #     regression_stat = fit_regression(prob_pre, diff)
    #     stats['regression'] = regression_stat
    return stats , counts, result
    
def fit_regression(prob_pre, diff):
    
    mask = (prob_pre >= 0.1) & (prob_pre <= 0.9)        
    filtered_diff = diff[mask]
    # Convert to numpy for sklearn
    prob_pre_np = prob_pre[mask].cpu().numpy().reshape(-1, 1)
    diff_np = filtered_diff.cpu().numpy().reshape(-1, 1)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(prob_pre_np, diff_np)

    # Get the slope (alpha)
    alpha = model.coef_[0][0]
    # Get the intercept
    intercept = model.intercept_[0]

    # Predict the values
    diff_pred = model.predict(prob_pre_np)

    # Calculate R-squared
    r2 = r2_score(diff_np, diff_pred)

    # Calculate Mean Squared Error
    mse = mean_squared_error(diff_np, diff_pred)
    return {'alpha': alpha, 'intercept': intercept, 'r2': r2, 'mse': mse}
    
def load_pt(path):
    # print(f"Loading from {path}")
    data = torch.load(path)
    return data

def original_stats(tensor):
    
    ranges = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1)]
    result = {}
    for r_min, r_max in ranges:
        mask = (tensor >= r_min) & (tensor < r_max)        
        filtered_diff = tensor[mask]
        mean_orig = tensor[mask].mean().item()
        result[f"{r_min}-{r_max}"] = {
                'orig_mean': mean_orig,
                'orig_count': filtered_diff.shape[0],
        }
    for k,v in result.items():
        print(f"{k}|{v['orig_mean']}|{v['orig_count']}")
    
def compare_prob():
    steps = [40000, 110000, 194000, 278000, 362000, 432000, 502000, 556000]
    next_steps = [41000, 111000, 195000, 279000, 363000, 432410, 503000, 557000]
    for step, next_step in zip(steps, next_steps):
        prob_next_0 = load_pt(f"checkpoints/pretrained/{step}/gold_probabilities_manual_1k{int(step/1000)}.pt")
        prob_next_1 = load_pt(f"checkpoints/pretrained/{next_step}/gold_probabilities_manual_1k{int(step/1000)}.pt")
        # import pdb; pdb.set_trace()
        stats, _, result = analyse(prob_next_0, prob_next_1)
        print("-"*50)
        print(f"\n{step} for next 1k")
        
        # print("|".join(stats['regression'].keys()))
        # print("|".join([str(v) for v in stats['regression'].values()]))
        for k,v in result.items():
            print(f"{k}|{v['orig_mean']}|{v['orig_count']}|{v['mean']}|{v['std']}|{v['min']}|{v['max']}|{v['count']}")
                
        prob_prev_0 = load_pt(f"checkpoints/pretrained/{step}/gold_probabilities_manual_1k{int(step/1000 -1)}.pt")
        prob_prev_1 = load_pt(f"checkpoints/pretrained/{next_step}/gold_probabilities_manual_1k{int(step/1000 -1)}.pt")

        stats, _, result = analyse(prob_prev_0, prob_prev_1)
        print("\n\n")
        # print("|".join(stats['regression'].keys()))
        # print("|".join([str(v) for v in stats['regression'].values()]))
        for k,v in result.items():
            print(f"{k}|{v['orig_mean']}|{v['orig_count']}|{v['mean']}|{v['std']}|{v['min']}|{v['max']}|{v['count']}")

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
        # import pdb; pdb.set_trace()
        for k in data.keys():
            if isinstance(data[k], list):
                for layer_idx in range(len(data[k])):
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
            
        # total_variance = torch.var(all_gold_probabilities, unbiased=False)
        # result["pred_distribution_variance"].append(str(total_variance.item()))

    print(step_temp)
    result = dict(sorted(result.items()))
    for k,v in result.items():
        if "instances" not in k:
            print(f"{k}|{'|'.join(v)}")

    print("\n\n\n", step_temp_prob)
    result_prob = dict(sorted(result_prob.items()))
    for k,v in result_prob.items():
        print(f"{k}|{'|'.join(v)}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--report", type=bool, default=False) #if True, pretrained report // False, finetuned report
    parser.add_argument("--data_type", type=str, default="last_1k")
    args = parser.parse_args()
    
    
    if args.report:
        report(args.data_type)
        # compare_prob()
    else:
        main(args)
        
    