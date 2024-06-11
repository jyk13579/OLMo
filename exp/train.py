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
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import argparse
from dataset import CustomDataset

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

# @hydra.main(version_base=None, config_path="config")
# def main(config: DictConfig) -> None:

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

    # def tokenize_func(eg):
    #     result = tokenizer(
    #         eg["text"],
    #         # eg["text"] if "text" in eg else eg["MedlineCitation"]["Article"]["Abstract"]["AbstractText"],       # extract abstract text from pubmed corpus
    #         truncation=True,
    #         max_length=config.max_token_length,
    #         padding=False,
    #         return_tensors=None,
    #     )
    #     return result
    def tokenize_func(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_token_length,
            padding=False,
        )
                                              
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
            # dataset = load_dataset(config.dataset)[config.split]
            # if config.subset_examples is not None:
            #     dataset = dataset.shuffle(seed=config.seed).select(range(config.subset_examples))
            # dataset = dataset.map(
            #     tokenize_func, 
            #     remove_columns=list(set(dataset.column_names) - set(inspect.signature(model.forward).parameters.keys()))
            # )
        else:
            # dataset = read_json_file(f"data/corpus/{config.dataset}.json")
            # print("Start tokenize function")
            # dataset = list(map(tokenize_func, dataset))
            # print("Done with tokenize_func")
            
            print("\n Loading data")
            custom_dataset = CustomDataset(tokenizer, config)
            print("\n Loading data DONE")
            custom_dataset.print_sample()
            # data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)

    print("Prepare Trainer")
    # Prepare trainer
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
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
            evaluation_strategy="no",
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            output_dir=config.save_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            report_to="none",
            seed=config.seed,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8,
        ),
    )

    # Perform training or evaluation
    if config.mode == 'train':
        # trainer.train_dataset = dataset
        trainer.train_dataset = custom_dataset
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args)
