import json
from transformers import Trainer, EvalPrediction, TrainerCallback
import torch 
import torch.distributed as dist
from dataset import CustomDataset, SlotDataset
from transformers import DataCollatorForLanguageModeling
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res
def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Wrote json file to: {file_path}!")
    
class OnTrainBeginCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """
    def on_train_begin(self, args, state, control, **kwargs):
        control.should_training_stop = False
        control.should_evaluate = True
        
            
class ExpTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.compute_metrics_defined
        if "fictional" in self.args.output_dir: 
            data = read_json_file("data/corpus/fictional/fictional_keyword.json") 
            self.slot_data = [d for d in data if d['type']=='paraphrase_gen']
        else:
            self.slot_data = read_json_file("data/corpus/pubmed_keyword.json")
            
    def compute_metrics_defined(self, prediction: EvalPrediction, metric_key_prefix, loss):
        gpu_num = self.args.process_index
        metric = {}
        # import pdb; pdb.set_trace()
        if 'original' in metric_key_prefix:
            if gpu_num <= 1:
                dataset_prob_eval = self.eval_dataset['original'] if gpu_num == 0 else self.eval_dataset['dolma_prev']
                name_ = "pubmed_orig" if gpu_num == 0 else "dolma_prev1k"
                eval_dataloader = self.get_eval_gen_dataloader(dataset_prob_eval, batch_size = 2)
                all_gold_probabilities = []
                all_mask = []
                
                with torch.inference_mode():
                    for batch in tqdm(eval_dataloader):
                        input_ids = batch['input_ids'].to(self.args.device)
                        outputs = self.model(input_ids=input_ids, output_attentions=True)
                        
                        logits = outputs.logits.detach()                        
                        logits = logits[:, :-1, :]                    
                        logits = logits.contiguous() 
                        shift_labels = input_ids[..., 1:].contiguous() 

                        shift_probabilities = torch.softmax(logits, dim=-1)       
                        gold_probabilities = torch.gather(shift_probabilities, -1, shift_labels.unsqueeze(-1)).squeeze(-1).detach().cpu()
                        all_gold_probabilities.append(gold_probabilities)
                        mask = (shift_labels != 1)
                        all_mask.append(mask)
                      
                
                all_gold_probabilities = torch.cat(all_gold_probabilities, dim=0).cpu()
                all_mask = torch.cat(all_mask, dim=0).cpu()
                all_gold_probabilities = all_gold_probabilities[all_mask]
                mean_probability = all_gold_probabilities.mean().item()
                median_probability = all_gold_probabilities.median().item()
                variance_probability = torch.var(all_gold_probabilities, unbiased=False).item()
                
                metric[f"{name_}_mean_probability"] = mean_probability
                metric[f"{name_}_median_probability"] = median_probability
                metric[f"{name_}_variance_probability"] = variance_probability
                
                all_gold_probabilities = all_gold_probabilities.float().half().detach()
                torch.save(all_gold_probabilities, f"{self.args.output_dir}/step_{self.state.global_step}_gold_probabilities_{name_}.pt")
            elif gpu_num <= 4:
                dataset_slot_gen = self.eval_dataset['slot_gen_orig'] if gpu_num == 2 else self.eval_dataset['slot_gen_para']
                name_ = "slot_generation_original" if gpu_num == 2 else "slot_generation_paraphrase"
                gt_dataset = self.slot_data
                ground_truths = [gt['answer'] for gt in gt_dataset]               
                bs = 4 #self.args.per_device_eval_batch_size
                eval_dataloader = self.get_eval_gen_dataloader(dataset_slot_gen, batch_size = bs)
                output_to_save = []
                generate_has_answers = []
                for batch_idx, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                    input_ids = batch["input_ids"]
                    label_ids = batch["labels"]
                    attention_mask = batch["attention_mask"]
                    # import pdb; pdb.set_trace()
                        
                    non_padded_lengths = attention_mask.sum(dim=1)
                    valid_labels_mask = label_ids != -100
                    valid_label_sums = valid_labels_mask.sum(dim=1)
                    difference = non_padded_lengths - valid_label_sums
                    
                    max_length = max(difference)
                    padded_input_ids = torch.stack([
                        torch.cat([
                            torch.full((max_length - dif.item(),), self.tokenizer.pad_token_id, dtype=ids.dtype), 
                            ids[:dif.item()]  # Valid input IDs shifted to the right
                        ]) for ids, dif in zip(input_ids, difference)
                    ]).to(self.args.device)

                    new_attention_mask = torch.stack([
                        torch.cat([
                            torch.zeros(max_length - dif.item(), dtype=torch.long), 
                            torch.ones(dif.item(), dtype=torch.long) 
                        ]) for dif in difference
                    ]).to(self.args.device)
                        
                    num_return_sequences = 5
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            input_ids = padded_input_ids,
                            attention_mask = new_attention_mask,
                            do_sample=True, 
                            temperature=0.8, 
                            num_beams=5, 
                            max_new_tokens=20, 
                            min_length=3, 
                            num_return_sequences=num_return_sequences, 
                            use_cache=True)
                        
                    output_sequences_ = output_ids.view(padded_input_ids.shape[0], num_return_sequences, -1) 
                    input_sequences = self.tokenizer.batch_decode(padded_input_ids, skip_special_tokens=True)
            
                    for inbatch_id, output in enumerate(output_sequences_):
                        generated_answer = self.tokenizer.batch_decode(output[:, max_length:], skip_special_tokens=True)
                        
                        index = bs * batch_idx + inbatch_id
                        this_answer = ground_truths[index]

                        if isinstance(this_answer, str):
                            has_answer = [this_answer.lower() in text.lower() for text in generated_answer]
                        elif isinstance(this_answer, list):
                            has_answer = [any(ans.lower() in output_item.lower() for ans in this_answer) for output_item in generated_answer]
                            
                        generate_has_answers.append(sum(has_answer)>0)
                        # print(gt_dataset[index])
                        output_to_save.append({
                            "question_id": gt_dataset[index]['id'],
                            "input": input_sequences[inbatch_id],
                            "generated_text": generated_answer,
                            "answer": gt_dataset[index]['answer'],
                            "has_answer": has_answer
                        })
                            
                write_json_file(f"{self.args.output_dir}/step_{self.state.global_step}_gold_probabilities_{name_}.json", output_to_save)
                eval_generate_accuracy = sum(generate_has_answers) / len(generate_has_answers)
                perplexity = np.exp(loss)
                metric[f"{name_}_gen_accuracy"] = eval_generate_accuracy
                metric["dummy_0"] = 0.0
                metric["dummy_1"] = 0.0
            else:   
                pass
            
            # print(f"Done for {name_}")
            # print(metric)
            
            # Prepare for gathering: Convert dict to a tensor for each key-value pair
            # This assumes all values are scalar and can be represented as float
            metric_tensor = torch.tensor([value for value in metric.values()], device=self.args.device)
            world_size = torch.distributed.get_world_size()
            
            # Gather all metric tensors on each GPU
            gathered_metrics = [torch.zeros_like(metric_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_metrics, metric_tensor)

            # print(gpu_num, "-- metric_tensor: ", metric_tensor)
            # Only process gathered metrics on master GPU
            if gpu_num == 0:
                # Convert list of tensors back into a dictionary
                # print("gathered_metrics")
                # print(gathered_metrics)
                keys = [["pubmed_orig_mean_probability", "pubmed_orig_median_probability", "pubmed_orig_variance_probability"], ["dolma_prev1k_mean_probability", "dolma_prev1k_median_probability", "dolma_prev1k_variance_probability"], ["slot_generation_original_gen_accuracy", "dummy1", "dummy2"], ["slot_generation_paraphrase_gen_accuracy", "dummy3", "dummy4"]]  
                            
                agg_metrics = {}
                for j in range(4):
                    for i in range(3):
                        if "dummy" not in keys[j][i]:
                            agg_metrics[keys[j][i]] = gathered_metrics[j][i].item()
                # print("\n\naggregated_metrics",agg_metrics,"\n","-"*50)
                            
                metric = agg_metrics
        else:
            perplexity = np.exp(loss)
            metric["perplexity"]=perplexity
            
        dist.barrier()
        return metric
    
    
    def get_eval_gen_dataloader(self, eval_dataset: Optional[Dataset] = None, batch_size = 1) -> DataLoader:
        
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(eval_dataset, **dataloader_params)
    
    
# def aggregate_metrics(metric_dicts):
#     # Example aggregation by averaging
#     aggregated = {}
#     print("!!!!!!!!!!!", metric_dicts)
#     for metric_each in metric_dicts:
        
#     for key in metric_dicts[0].keys():
#         aggregated[key] = sum(d[key] for d in metric_dicts) / len(metric_dicts)
#     return aggregated

from dataclasses import dataclass, field
import transformers
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances) :
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch