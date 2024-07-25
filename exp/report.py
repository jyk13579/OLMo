import json
from collections import defaultdict 
import argparse
import glob
import torch
import os
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
LOG_PROB = [f"eval_new_log_prob_diff_{k}" for k in ranges] + [f"eval_prev_log_prob_diff_{k}" for k in ranges] 
                
EVAL_5 = EVAL_5_LOSS + EVAL_5_PROB

def main(args):
    
    paths = [f"{args.folder_path}/{n}" for n in os.listdir(args.folder_path)] if args.folder_path is not None else [args.path]
    # path = args.path
    for path in paths:
        result = defaultdict(dict)
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
        result = analyse_prob(path, result, log_prob=True)
        
        # keys = EVAL_1 + EVAL_2 + EVAL_3 + EVAL_4 + EVAL_5
        # keys = EVAL_4_PROB + EVAL_5_PROB
        keys = LOG_PROB
        mod = str(int(path.split("/")[-1].split("_")[0])//1000)+"k"
        epo = path.split("/")[-1].split("_")[3].replace("ep","")
        lr = path.split("/")[-1].split("_")[2].replace("e", "e-")
        
        # print("\n\n", "-"*50, path)
        for key in keys:
            # print_values(result[key], key)
            print_values(result[key], f"{lr}|{epo}|{mod}|{key}")
    
def print_values(res, key):
    print(f"{key}|", "|".join(res.values()))
    
def print_prob(path): 
    result = analyse_prob(path)
    keys = EVAL_4_PROB + EVAL_5_PROB
    for key in keys:
        print_values(result[key], key)
        
def analyse_prob(path, result=defaultdict(dict), log_prob = False): 
        
    sorted_loaded_files = load_tensors(path, "pubmed_orig")
    for ep in range(len(sorted_loaded_files)):
        stats, counts, _ = analyse(sorted_loaded_files[0], log_prob = log_prob) if ep == 0 else analyse(sorted_loaded_files[0], sorted_loaded_files[ep], log_prob = log_prob)
        # import pdb; pdb.set_trace()
        for k,v in stats.items():
            result[f"eval_new_{'log_' if log_prob else ''}prob_diff_{k}"][ep] = str(v)
            result[f"eval_new_{'log_' if log_prob else ''}prob_count_{k}"][ep] = str(counts[k])
        
    sorted_loaded_files = load_tensors(path, "dolma_prev1k")
    for ep in range(len(sorted_loaded_files)):
        stats, counts, _ = analyse(sorted_loaded_files[0], log_prob = log_prob) if ep == 0 else analyse(sorted_loaded_files[0], sorted_loaded_files[ep], log_prob = log_prob)
        for k,v in stats.items():
            result[f"eval_prev_{'log_' if log_prob else ''}prob_diff_{k}"][ep] = str(v)
            result[f"eval_prev_{'log_' if log_prob else ''}prob_count_{k}"][ep] = str(counts[k])
    
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
    
def analyse(prob_pre, prob_ft=None, ranges=custom_ranges, log_prob = False):
    stats = {}
    counts = {}
    result = {}
    # import pdb; pdb.set_trace()
    if prob_ft is not None:
        if log_prob:
            diff = -torch.log(prob_ft+ 1e-6) + torch.log(prob_pre+ 1e-6)
            log_prob_pre = -torch.log(prob_pre + 1e-6)
        else:
            diff = prob_ft - prob_pre
    else:
        if log_prob:
            diff = -torch.log(prob_pre+ 1e-6)
        else:
            diff = prob_pre
    stats['all'] = diff.mean().item()
    counts['all'] = diff.shape[0]
    for r_min, r_max in ranges:
        mask = (prob_pre >= r_min) & (prob_pre < r_max)        
        filtered_diff = diff[mask]
        if filtered_diff.numel() > 0:
            mean_orig = log_prob_pre[mask].mean().item() if log_prob else prob_pre[mask].mean().item()
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
                'orig_portion': filtered_diff.shape[0]/diff.numel(),
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
        stats, _, result = analyse(prob_next_0, prob_next_1, log_prob=True)
        print("-"*50)
        print(f"{step} for next 1k")
        
        # print("|".join(stats['regression'].keys()))
        # print("|".join([str(v) for v in stats['regression'].values()]))
        for k,v in result.items():
            print(f"{k}|{v['orig_mean']}|{v['orig_count']}|{v['orig_portion']}|{v['mean']}|{v['std']}|{v['min']}|{v['max']}|{v['count']}")
        print("")
    for step, next_step in zip(steps, next_steps):
        prob_prev_0 = load_pt(f"checkpoints/pretrained/{step}/gold_probabilities_manual_1k{int(step/1000 -1)}.pt")
        prob_prev_1 = load_pt(f"checkpoints/pretrained/{next_step}/gold_probabilities_manual_1k{int(step/1000 -1)}.pt")

        stats, _, result = analyse(prob_prev_0, prob_prev_1, log_prob=True)
        print("-"*50)
        print(f"{step} for previous 1k")
        # print("|".join(stats['regression'].keys()))
        # print("|".join([str(v) for v in stats['regression'].values()]))
        for k,v in result.items():
            print(f"{k}|{v['orig_mean']}|{v['orig_count']}|{v['orig_portion']}|{v['mean']}|{v['std']}|{v['min']}|{v['max']}|{v['count']}")
        print("")
        
def report(data_type):
    import os
    if 'dolma' in data_type:
        filelist = [5000, 110000, 194000, 278000, 362000, 432410, 502000, 557000]
        # filelist = [40000, 50000, 60000, 70000, 80000, 90000, 100000]
        eval_path_ = "checkpoints/pretrained/dolma_prob/{}/model{}_entropy_{}_new.json"
        prob_path_ = "checkpoints/pretrained/dolma_prob/{}/model{}_gold_prob_{}_new.pt"
        
    else:
        filelist = [int(n) for n in os.listdir("checkpoints/pretrained") if "_" not in n]
        eval_path_ = "checkpoints/pretrained/{}/entropy_{}.json"
        prob_path_ = "checkpoints/pretrained/{}/gold_probabilities_{}.pt"
        
    result = defaultdict(list)
    step_temp = "step|"
    result_prob = defaultdict(list)
    step_temp_prob = "step|"
    for step in sorted(filelist):
        if 'dolma' in data_type:
            eval_path = eval_path_.format(data_type, step, data_type)
            prob_path = prob_path_.format(data_type, step, data_type)
        else:
            eval_path = eval_path_.format(step, data_type)
            prob_path = prob_path_.format(step, data_type)
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
        # import pdb; pdb.set_trace()
        # gold prob
        if not os.path.isfile(prob_path):
            print(f"No file for {prob_path}.")
            continue
        all_gold_probabilities = torch.load(prob_path).cpu()
        all_gold_probabilities = all_gold_probabilities.float()
        log_prob = -torch.log(all_gold_probabilities+ 1e-6)
        mean_log = torch.mean(log_prob)
        result["pred_distribution_mean_log_prob"].append(str(mean_log.item()))
        # num_bins = 100
        # hist = torch.histc(all_gold_probabilities, bins=num_bins, min=0, max=1)
        # step_temp_prob += f"{step}|"
        # for idx, element in enumerate(hist):
        #     result_prob[f"prob_{idx/100}"].append(str(int(element.item())))
        
        if "new" in data_type or "cpt" in data_type:
            act_path = f"checkpoints/pretrained/{step}/mlp_activation_sparsity_raw_{data_type}.pt"
            if not os.path.isfile(act_path):
                print(f"No file for {act_path}.")
                continue
            mlp_act = torch.load(act_path).cpu()
            # import pdb; pdb.set_trace()
            for k in [11008, 3000, 1000, 300, 100]:
                avg, entropy = mlp_act_sparsity(mlp_act, k)
                result[f"entropy_act_sparsity_top{str(k)}"].append(str(avg))
                for layer_idx in range(32):
                    result[f"entropy_act_sparsity_top{str(k)}_layer_{str(layer_idx)}"].append(str(entropy[layer_idx].item()))
        # total_variance = torch.var(all_gold_probabilities, unbiased=False)
        # result["pred_distribution_variance"].append(str(total_variance.item()))

    print(step_temp)
    result = dict(sorted(result.items()))
    for k,v in result.items():
        if "instances" not in k:
            print(f"{k}|{'|'.join(v)}")

    # print("\n\n\n", step_temp_prob)
    # result_prob = dict(sorted(result_prob.items()))
    # for k,v in result_prob.items():
    #     print(f"{k}|{'|'.join(v)}")
        
def mlp_act_sparsity(act_sparsity, k):
    top_values, _ = torch.topk(act_sparsity, k=k, dim=1)
    prob_dist = top_values / torch.sum(top_values, dim=1, keepdim=True)
    entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10), dim=-1) #(32,)
    avg = entropy.sum().item()/32
    # print(entropy.sum()/32)
    # print(entropy)
    return avg, entropy

def initial_temperature(mlp):
    mlp = torch.load("checkpoints/pretrained/557000/mlp_activation_sparsity_raw_next_1k_new.pt")
    # for i in range(32):
    #     max_ = mlp[i].max().item()
    #     avg_ = mlp[i].sum().item()/11008
    #     top_values, _ = torch.topk(mlp_110k[i], k=100)
    #     min_top100 = top_values.min().item()
    #     print(max_, "|", avg_, "|", min_top100)
    new_tensor = torch.ones_like(mlp)
    # for i in range(mlp.size(0)):
    #     top_values, top_indices = torch.topk(mlp[i], 100)
    #     min_top_value = top_values.min()
    #     ratios = top_values / min_top_value
    #     new_tensor[i, top_indices] = ratios
    # torch.save(new_tensor, f"checkpoints/pretrained/557000/mlp_activation/top100_min_ratio.pt")
        
    mask = mlp > mlp.mean(dim=-1).unsqueeze(-1)
    # new_tensor[mask] = 2.0
    # torch.save(new_tensor, f"checkpoints/pretrained/557000/mlp_activation/top100_2.0.pt")
                                  
    mlp_110 = torch.load("checkpoints/pretrained/110000/mlp_activation_sparsity_raw_next_1k_new.pt")
    relative = mlp / mlp_110
    new_tensor[mask] = relative[mask]
    torch.save(new_tensor, f"checkpoints/pretrained/557000/mlp_activation/relative_to_110_uppermean.pt")
    torch.save(relative, f"checkpoints/pretrained/557000/mlp_activation/relative_to_110_all.pt")
    
    mlp = torch.load("checkpoints/pretrained/557000/mlp_activation_sparsity_raw_cpt_pubmed.pt")
    mlp_110 = torch.load("checkpoints/pretrained/110000/mlp_activation_sparsity_raw_cpt_pubmed.pt")
    mask = mlp > mlp.mean(dim=-1).unsqueeze(-1)
    relative = mlp / mlp_110
    new_tensor = torch.ones_like(mlp)
    new_tensor[mask] = relative[mask]
    torch.save(new_tensor, f"checkpoints/pretrained/557000/mlp_activation/relative_to_110_uppermean_pubmed.pt")

def calculate_auc():

    # Data from the table
    data_new = {
        '5k': [0.316, 0.499, 0.851, 0.962, 0.983, 0.989, 0.990, 0.991, 0.991, 0.991],
        '110k': [0.388, 0.569, 0.835, 0.935, 0.963, 0.983, 0.990, 0.991, 0.991, 0.991],
        '194k': [0.384, 0.543, 0.795, 0.922, 0.959, 0.979, 0.988, 0.991, 0.991, 0.991],
        '278k': [0.384, 0.537, 0.780, 0.914, 0.957, 0.977, 0.988, 0.991, 0.991, 0.991],
        '362k': [0.375, 0.504, 0.761, 0.911, 0.956, 0.976, 0.988, 0.991, 0.992, 0.992],
        '432k': [0.360, 0.463, 0.334, 0.821, 0.930, 0.968, 0.985, 0.990, 0.991, 0.991],
        '502k': [0.315, 0.393, 0.575, 0.837, 0.936, 0.968, 0.984, 0.989, 0.991, 0.991],
        '557k': [0.195, 0.331, 0.464, 0.746, 0.925, 0.968, 0.985, 0.990, 0.991, 0.991]
    }

    # Epochs
    import numpy as np
    epochs = np.arange(1, 11)

    # Calculate AUC for each key in the data dictionary
    auc_values = {}
    for key, values in data_new.items():
        auc = np.trapz(values, epochs)
        auc_values[key] = auc

        # Display the AUC values
    for key, auc in auc_values.items():
        print(f"{key}| {auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--folder_path", type=str, default=None)
    parser.add_argument("--report", type=bool, default=False) #if True, pretrained report // False, finetuned report
    parser.add_argument("--compare_prob", type=bool, default=False)
    parser.add_argument("--data_type", type=str, default="last_1k")
    args = parser.parse_args()
    
    
    if args.report:
        report(args.data_type)
    elif args.compare_prob:
        compare_prob()
    else:
        main(args)
        
    