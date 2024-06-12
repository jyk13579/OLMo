import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"]="OLMO"
import torch
import inspect
import transformers
# import hydra
import yaml
from omegaconf import OmegaConf as om
from omegaconf import DictConfig, OmegaConf
from transformers import OlmoForCausalLM, AutoTokenizer
from accelerate import Accelerator
# from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import argparse
from dataset import CustomDataset, IndexedDataset, read_json_file
from trainer import ExpTrainer, OnTrainBeginCallback
import numpy as np
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from dataclasses import dataclass, field

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            if key == 'lr':
                value = float(value)
            setattr(self, key, value)

# Load the YAML file
def load_config(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return Config(data)

def make_eval_data_module(tokenizer, config):
    slot_factual = ""
    slot_paraphrase = ""
    prob = ""
    log_diff = ""
    
    data = read_json_file("data/corpus/pubmed_derived.json")
    pubmed = CustomDataset(tokenizer, config, data=[d for d in data if d['type']=='pubmed'])
    casual = CustomDataset(tokenizer, config, data=[d for d in data if d['type']=='casual'])
    counterfactual = CustomDataset(tokenizer, config, data=[d for d in data if d['type']=='counterfactual'])
    
    evaluate_dataset = {
        'original': pubmed,
        'casual': casual,
        'counterfactual': counterfactual
    }
    
    return evaluate_dataset

class CustomTrainingArguments(transformers.TrainingArguments):
    def __init__(self, *args, step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = step
        
def main(args) -> None:
    config = load_config(args.config)

    # Prepare model
    model = OlmoForCausalLM.from_pretrained(f"checkpoints/pretrained/{config.step}/hf",
                                               attn_implementation="eager",
                                               torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    
    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id          
                                              
    with Accelerator().main_process_first():
        if config.split != "train":
            dataset = load_dataset(config.dataset)
            processed_datasets = {}
            for split, data in dataset.items():
                if config.subset_examples is not None:
                    data = data.shuffle(seed=config["seed"]).select(range(config["subset_examples"]))
                processed_datasets[split] = data.map(
                    tokenize_func,
                    batched=True,  # Assuming your function can handle batching
                    remove_columns=list(set(data.column_names) - set(inspect.signature(model.forward).parameters.keys())),
                    num_proc=4  # Adjust based on your available CPUs, optional but can speed up processing
                )
        else:
            print("\n Loading data")
            custom_dataset = CustomDataset(tokenizer, config)
            print("\n Loading data DONE")
            custom_dataset.print_sample()
            
            eval_dataset = make_eval_data_module(tokenizer, config)

    print("Prepare Trainer")
    # Prepare trainer
    # callbacks = []
    # callbacks.append(OnTrainBeginCallback())
    trainer = ExpTrainer(
        model=model,
        args=CustomTrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.accumulate_grad_batches,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.max_epochs,
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=config.lr,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.scheduler,
            max_grad_norm=config.gradient_clip_val,
            bf16=True,                
            logging_steps=1,
            evaluation_strategy="epoch",
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            output_dir=config.save_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            report_to="none",
            seed=config.seed,
            step=config.step,
            prediction_loss_only=True,
            include_inputs_for_metrics=True
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8,
        ),
        callbacks = [OnTrainBeginCallback()]
    )

    # Perform training or evaluation
    if config.mode == 'train':
        # trainer.train_dataset = dataset
        trainer.train_dataset = custom_dataset
        trainer.eval_dataset = eval_dataset
        print("Trainer ready")
        trainer.train()
        trainer.save_state()
        trainer.save_model(config.save_dir)

    elif config.mode == 'eval':
        for key, dataset_spl in processed_datasets.items():
            loss = trainer.evaluate(dataset_spl)['eval_loss']
            print(f'Loss on {key}: {loss:.4f}')

    else:
        raise ValueError("Invalid mode: %s" % config.mode)

def save_dolma(loc, step):
            
    if loc == "prev_1k":
        model_path = f"checkpoints/pretrained/{step}"
        train_state = torch.load(f"{model_path}/train.pt")
        
        if step <= 432410:
            epoch = 1
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen']
        else:
            epoch = 1
            global_train_examples_seen_this_epoch = train_state['global_train_examples_seen_this_epoch']
            
        global_train_examples_seen_this_epoch -= 2160000
    elif loc == "first_1k":
        epoch = 1
        global_train_examples_seen_this_epoch = 0
        
    data_order_file_path=f"data/global_indices/global_indices_epoch{epoch}.npy"
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    print(f"\n Loaded dataset \n epoch: {epoch} \n global_train_examples_seen_this_epoch : {global_train_examples_seen_this_epoch}")
    
    instances = []
    batch_start = global_train_examples_seen_this_epoch
    for i in range(1000):
        instances.append(global_indices[batch_start+i*2160])
    import pdb; pdb.set_trace()
    train_config_path = "configs/official/OLMo-7B_2160.yaml"    
    cfg = TrainConfig.load(train_config_path)
    dataset = build_memmap_dataset(cfg, cfg.data)
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
    to_save = []
    
    for inst in instances:
        input_ids = dataset[inst]
        text = tokenizer.batch_decode([input_ids])
        to_save.append({
            "id": inst,
            "text": text[0]
        })
    
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default=None)
    # args = parser.parse_args()
    # main(args)
    
    save_dolma('prev_1k', 556000)
