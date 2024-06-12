import json
from transformers import Trainer, EvalPrediction, TrainerCallback
import torch 
import numpy as np
from olmo.data import build_memmap_dataset
from olmo.config import TrainConfig
from dataset import IndexedDataset
from transformers import DataCollatorForLanguageModeling
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
        
def ecp(loc, step):
    if loc == "prev_1k":
        model_path = f"checkpoints/pretrained/{step}"
        train_state = torch.load(f"{model_path}/train.pt")
        
        epoch = 1 if step < 432410 else 2
        global_train_examples_seen_this_epoch = train_state.get("global_train_examples_seen_this_epoch", train_state['global_train_examples_seen'])
        global_train_examples_seen_this_epoch -= 2160
        if step == 432410:
            global_train_examples_seen_this_epoch = 0 
    elif loc == "first_1k":
        epoch = 1
        global_train_examples_seen_this_epoch = 0
        
    data_order_file_path=f"data/global_indices/global_indices_epoch{epoch}.npy"
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    print(f"\n Loaded dataset \n epoch: {epoch} \n global_train_examples_seen_this_epoch : {global_train_examples_seen_this_epoch}")
    
    instances = []
    batch_start = global_train_examples_seen_this_epoch
    for i in range(1000):
        instances.append(global_indices[batch_start+i])
        
    return instances
            
class ExpTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.compute_metrics_defined
        # self.test_data = read_json_file(f"{self.args.data_path}/test.jsonl")
        # self.data_dict = {
        #     "test": self.test_data,
        # }
        
        # train_config_path = "configs/official/OLMo-7B_2160.yaml"    
        # cfg = TrainConfig.load(train_config_path)
        # dataset = build_memmap_dataset(cfg, cfg.data)
        # dolma_prev = IndexedDataset(dataset, ecp("prev_1k", self.args.step))
        # dolma_start = IndexedDataset(dataset, ecp("first_1k"))
    
            
    def compute_metrics_defined(self, prediction: EvalPrediction, metric_key_prefix, loss):
        import pdb; pdb.set_trace()
        metric = {}
        if 'original' in metric_key_prefix:
            if self.args.process_index == 0:
                dataset_original = self.eval_dataset['original']
                eval_dataloader = self.get_eval_gen_dataloader(dataset_original, batch_size = self.args.per_device_eval_batch_size)
                all_gold_probabilities = []
                all_mask = []
                
                with torch.inference_mode():
                    for batch in tqdm(eval_dataloader):
                        input_ids = batch['input_ids'].to(self.args.device)
                        outputs = self.model(input_ids=input_ids, output_attentions=True)
                        
                        # entropy of prediction prob
                        logits = outputs.logits
                        
                        logits = logits[:, :-1, :]
                    
                        logits = logits.contiguous() 
                        shift_labels = input_ids[..., 1:].contiguous() 

                        shift_probabilities = torch.softmax(logits, dim=-1)       
                        gold_probabilities = torch.gather(shift_probabilities, -1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()
                        all_gold_probabilities.append(gold_probabilities)
                        mask = (shift_labels != 1)
                        all_mask.append(mask)
                
                all_gold_probabilities = torch.cat(all_gold_probabilities, dim=0)
                all_mask = torch.cat(all_mask, dim=0)
                all_gold_probabilities = all_gold_probabilities[all_mask]
                mean_probability = all_gold_probabilities.mean().item()
                median_probability = all_gold_probabilities.median().item()
                variance_probability = torch.var(all_gold_probabilities, unbiased=False).item()
                
                metric = {"mean_probability": mean_probability, 
                            "median_probability": median_probability,
                            "variance_probability": variance_probability,
                        }
                all_gold_probabilities = all_gold_probabilities.float().half().detach().cpu()
                torch.save(all_gold_probabilities, f"{self.args.output_dir}/step_{self.state.global_step}_gold_probabilities_original.pt")
    
        # test_name = metric_key_prefix.replace("eval_","")
        # if test_name in self.data_dict:
        #     to_test_dataset = self.eval_dataset[test_name]
        #     gt_dataset = self.data_dict[test_name]
        #     ground_truths = [gt['answer'] for gt in gt_dataset]
        #     # import pdb; pdb.set_trace()
        #     eval_generate_accuracy = None
        #     perplexity = None
        #     if self.args.process_index == 0 and getattr(self.args, 'generate_eval', False):
        #         eval_dataloader = self.get_eval_gen_dataloader(to_test_dataset, batch_size = self.args.per_device_eval_batch_size)
        #         output_to_save = []
        #         generate_has_answers = []
        #         for batch_idx, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        #             input_ids = batch["input_ids"]
        #             label_ids = batch["labels"]
        #             attention_mask = batch["attention_mask"]
                    
        #             non_padded_lengths = attention_mask.sum(dim=1)
        #             valid_labels_mask = label_ids != -100
        #             valid_label_sums = valid_labels_mask.sum(dim=1)
        #             difference = non_padded_lengths - valid_label_sums
                    
        #             max_length = max(difference)
        #             padded_input_ids = torch.stack([
        #                 torch.cat([
        #                     torch.zeros(max_length - dif.item(), dtype=ids.dtype),  # Left padding
        #                     ids[:dif.item()]  # Valid input IDs shifted to the right
        #                 ]) for ids, dif in zip(input_ids, difference)
        #             ]).to(self.args.device)

        #             new_attention_mask = torch.stack([
        #                 torch.cat([
        #                     torch.zeros(max_length - dif.item(), dtype=torch.long), 
        #                     torch.ones(dif.item(), dtype=torch.long) 
        #                 ]) for dif in difference
        #             ]).to(self.args.device)
                    
        #             num_return_sequences = 1 if "mc" in self.args.data_path else 5
        #             with torch.inference_mode():
        #                 output_ids = self.model.generate(
        #                     input_ids = padded_input_ids,
        #                     attention_mask = new_attention_mask,
        #                     do_sample=True, 
        #                     temperature=0.8, 
        #                     num_beams=5, 
        #                     max_new_tokens=20, 
        #                     min_length=3, num_return_sequences=num_return_sequences, use_cache=True)
                    
                    
        #             output_sequences_ = output_ids.view(padded_input_ids.shape[0], num_return_sequences, -1) 
        #             # prompt_length = padded_input_ids.shape[1]
        #             input_sequences = self.tokenizer.batch_decode(padded_input_ids, skip_special_tokens=True)
            
        #             for inbatch_id, output in enumerate(output_sequences_):
        #                 generated_answer = self.tokenizer.batch_decode(output[:, max_length:], skip_special_tokens=True)
        #                 generated_answer = [text.split("#")[0] if "#" in text else text for text in generated_answer]
                        
        #                 index = self.args.per_device_eval_batch_size * batch_idx + inbatch_id
        #                 this_answer = ground_truths[index]

        #                 if isinstance(this_answer, str):
        #                     has_answer = [this_answer.lower() in text.lower() for text in generated_answer]
        #                 elif isinstance(this_answer, list):
        #                     has_answer = [any(ans.lower() in output_item.lower() for ans in this_answer) for output_item in generated_answer]
                        
        #                 generate_has_answers.append(sum(has_answer)>0)
        #                 # print(gt_dataset[index])
        #                 output_to_save.append({
        #                     "question_id": gt_dataset[index]['id'] if 'id' in gt_dataset[index] else gt_dataset[index]['docid'],
        #                     "input": gt_dataset[index]['input'] if 'input' in gt_dataset[index] else (
        #                             gt_dataset[index]['QA'] if 'QA' in gt_dataset[index] else input_sequences[inbatch_id]
        #                         ),
        #                     "generated_text": generated_answer,
        #                     "answer": gt_dataset[index]['answer'],
        #                     "has_answer": has_answer
        #                 })
                        
        #         write_jsonl_file(self.args.output_dir+f"/generated_output_{self.state.global_step}_{test_name}.jsonl", output_to_save)
        #         eval_generate_accuracy = sum(generate_has_answers) / len(generate_has_answers)
                    
        #         perplexity = np.exp(loss)

        #     metric = {"perplexity": perplexity, 
        #                 "gen_accuracy": eval_generate_accuracy,
        #                 "gen_accuracy_reverse": -1 * eval_generate_accuracy if eval_generate_accuracy is not None else None,
        #             }
                
        # else:
        #     perplexity = np.exp(loss)
        #     metric = {"perplexity": perplexity,}
            
        return metric
    
    def get_eval_gen_dataloader(self, eval_dataset: Optional[Dataset] = None, batch_size = 1) -> DataLoader:
        
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        # only if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset)
        eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

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