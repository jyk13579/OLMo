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
from modeling_olmo_hf import ExpOlmoForCausalLM
from accelerate import Accelerator
# from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import argparse
from dataset import CustomDataset, IndexedDataset, SlotDataset, read_json_file, write_json_file
from trainer import ExpTrainer, OnTrainBeginCallback, DataCollatorForSupervisedDataset
import numpy as np
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from dataclasses import dataclass, field
from tqdm import tqdm 

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            if key == 'lr':
                value = float(value)
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)
    
# Load the YAML file
def load_config(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return Config(data)

def make_eval_data_module(tokenizer, config):
    dolma_prev_data = read_json_file(f"data/dolma/step_{config.step}_prev_1k.json")
    if "pythia" in config.model:
        dolma_prev_data = read_json_file(f"data/dolma/thepile/random20480.json")[:1000]
        
    if config.dataset == 'pubmed':
        data = read_json_file("data/corpus/pubmed_derived.json")
        pubmed = CustomDataset(tokenizer, config, data=[d for d in data if d['type']=='pubmed'])
        casual = CustomDataset(tokenizer, config, data=[d for d in data if d['type']=='casual'])
        counterfactual = CustomDataset(tokenizer, config, data=[d for d in data if d['type']=='counterfactual'])
        
        original = SlotDataset(tokenizer, corpus_type="original", keywords_slot=True, config=config)
        paraphrase = SlotDataset(tokenizer, corpus_type="paraphrase", keywords_slot=True, config=config)
            
        dolma_prev = CustomDataset(tokenizer, data=dolma_prev_data, config=config)
        slot_gen_orig = SlotDataset(tokenizer, corpus_type="original",keywords_slot=False, config=config)
        slot_gen_para = SlotDataset(tokenizer, corpus_type="paraphrase", keywords_slot=False, config=config)
        
        evaluate_dataset = {
            'original': pubmed,
            'casual': casual,
            'counterfactual': counterfactual,
            'slotPubmed': original,
            'slotParaphrase': paraphrase,
            'dolma_prev': dolma_prev,
            'slot_gen_orig': slot_gen_orig,
            'slot_gen_para': slot_gen_para,
        }
    elif config.dataset == 'fictional':
        data = read_json_file("data/corpus/fictional/fictional_keyword.json")        
        original_corpus = CustomDataset(tokenizer, config, data=[d for d in data if d['type']=='original'])
        dolma_prev = CustomDataset(tokenizer, data=dolma_prev_data, config=config)
        
        original = SlotDataset(tokenizer, corpus_type="original", keywords_slot=True, config=config, data=[d for d in data if d['type']=='original'])
        paraphrase = SlotDataset(tokenizer, corpus_type="paraphrase", keywords_slot=True, config=config, data=[d for d in data if d['type']=='paraphrase'])
        slot_gen_orig = SlotDataset(tokenizer, corpus_type="original",keywords_slot=False, config=config, data=[d for d in data if d['type']=='original_gen'])
        slot_gen_para = SlotDataset(tokenizer, corpus_type="paraphrase", keywords_slot=False, config=config, data=[d for d in data if d['type']=='paraphrase_gen'])
        
        evaluate_dataset = {
            'original': original_corpus,
            'slotPubmed': original,
            'slotParaphrase': paraphrase,
            'dolma_prev': dolma_prev,
            'slot_gen_orig': slot_gen_orig,
            'slot_gen_para': slot_gen_para,
        }
    
    return evaluate_dataset

def num_parameters(module: torch.nn.Module, requires_grad: bool = None) -> int:
    return sum(p.numel() for p in module.parameters() if requires_grad is None or p.requires_grad == requires_grad)
def print_trainables(model):
    print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True)}")
    print("\n\n Trainable parameters")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n} : {p.shape}")
    print("\n\n")
    print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False)}")
    print("\n\n non Trainable parameters")
    for n, p in model.named_parameters():
        if not p.requires_grad :
            print(f"{n} : {p.shape}")
    print("\n\n")
    
class CustomTrainingArguments(transformers.TrainingArguments):
    def __init__(self, *args, step=0, initial_temp=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = step
        self.initial_temp = initial_temp
        self.final_temp = 1.0
        
def main(args) -> None:
    config = load_config(args.config)

    # Prepare model
    if "OLMo" in config.model:
        if "entropy" in config.save_dir:        
            # model = OlmoForCausalLM.from_pretrained(f"checkpoints/pretrained/{config.step}/hf", attn_implementation="eager")
            model = ExpOlmoForCausalLM.from_pretrained(f"checkpoints/pretrained/{config.step}/hf", attn_implementation="eager")            
            # from peft import LoraConfig, get_peft_model
            # lora_config = LoraConfig(
            #     r=4,
            #     lora_alpha=40,
            #     target_modules='.*(gate_proj|up_proj|down_proj)$',
            #     task_type="CAUSAL_LM",
            # )
            # print("Adding LoRA adapters...")
            # model = get_peft_model(model, lora_config)
        else:
            if config.step >0 :
                model = OlmoForCausalLM.from_pretrained(f"checkpoints/pretrained/{config.step}/hf", attn_implementation="eager")
            else:
                model = OlmoForCausalLM.from_pretrained("allenai/OLMo-7B-hf")
    elif "pythia" in config.model:        
        model = GPTNeoXForCausalLM.from_pretrained(
            config.model,
            revision="step143000",
            cache_dir=f"./checkpoints/pretrained/pythia/step143000",
            )
    model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    
    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id    
        tokenizer.pad_token = tokenizer.unk_token      
                                              
    with Accelerator().main_process_first():
        if config.split == "train":
            custom_dataset = CustomDataset(tokenizer, config)
            if torch.cuda.current_device() == 0:
                print("\n Loading data DONE")
                custom_dataset.print_sample()
            
            eval_dataset = make_eval_data_module(tokenizer, config)
        else:
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

    if torch.cuda.current_device() == 0:
        print("Prepare Trainer")
    # Prepare trainer
    callbacks = []
    # callbacks.append(OnTrainBeginCallback())
    print_trainables(model)
    # import pdb; pdb.set_trace()
    trainer = ExpTrainer(
        model=model,
        tokenizer=tokenizer,
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
            save_only_model=True,
            save_strategy=config.save_strategy,
            output_dir=config.save_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            report_to="wandb",
            seed=config.seed,
            step=config.step,
            prediction_loss_only=True,
            include_inputs_for_metrics=True,
            run_name = config.save_dir.split("/")[-1],
            initial_temp=config.get("initial_temp", 1.0)
        ),
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            # transformers.DataCollatorForLanguageModeling(
            #     tokenizer, mlm=False, pad_to_multiple_of=8,
            # ),
        callbacks = callbacks
    )

    # Perform training or evaluation
    if config.mode == 'train':
        # trainer.train_dataset = dataset
        trainer.train_dataset = custom_dataset
        trainer.eval_dataset = eval_dataset
        if torch.cuda.current_device() == 0:
            print("Trainer ready")
            print("Start Training")
        trainer.train()
        if torch.cuda.current_device() == 0:
            print("Training Done")
            print("Start Saving")
        trainer.save_state()
        if config.save_strategy == "no":
            trainer.save_model(config.save_dir)

    elif config.mode == 'eval':
        for key, dataset_spl in processed_datasets.items():
            loss = trainer.evaluate(dataset_spl)['eval_loss']
            print(f'Loss on {key}: {loss:.4f}')

    else:
        raise ValueError("Invalid mode: %s" % config.mode)

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
        
    data_order_file_path=f"data/global_indices/global_indices_epoch{epoch}.npy"
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    print(f"\n Loaded dataset \n epoch: {epoch} \n global_train_examples_seen_this_epoch : {global_train_examples_seen_this_epoch}")
    
    instances = []
    batch_start = global_train_examples_seen_this_epoch
    for i in range(1000):
        instances.append(global_indices[batch_start+i*2160])
        
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
        
    write_json_file(f"data/dolma/step_{step}_{loc}.json", to_save)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args)
    