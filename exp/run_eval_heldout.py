### Run again


import sys
sys.path.append("/home/hyeonbin/iterpre/OLMo")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
from tqdm.auto import tqdm
import argparse
from torch.nn import functional as F
import json

from olmo.checkpoint import load_state_dict
from olmo.eval import build_evaluators
from olmo.config import TrainConfig
from olmo.util import clean_opt
from olmo.model import OLMo

# Access the argument

NUM_DEVICES = 4

parser = argparse.ArgumentParser(description="")
parser.add_argument('--idx', type=int, help='GPU(Device) Number')
parser.add_argument("--ckpt_num", type=int, help='ckpt to run')
args = parser.parse_args()

# Access the argument
gpu_idx = args.idx
ckpt_num = args.ckpt_num

device = torch.device(f"cuda:{gpu_idx}")
result_path = f"/home/hyeonbin/iterpre/OLMo/olmo/exp/OLMo-7B_ckpt_{ckpt_num}_validation_set_ppl.jsonl"

yaml_path = "/home/hyeonbin/iterpre/OLMo/configs/official/OLMo-7B.yaml"
args_list = []
cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
cfg.device_eval_batch_size = 1
cfg.model.init_device = f"cuda:{gpu_idx}"

tok = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-hf", trust_remote_code=True)

# [Todo] Organize code later.
if ckpt_num == 5000:
    load_path = f"https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/step5000-unsharded/"
elif ckpt_num == 278000:
    load_path = f"https://olmo-checkpoints.org/ai2-llm/olmo-medium/yuc5kl7s/step278000-unsharded/"
elif ckpt_num == 557000:
    load_path = f"https://olmo-checkpoints.org/ai2-llm/olmo-medium/z4z0x4m9/step557000-unsharded/"
else:
    raise AssertionError("Not Implemented Yet!")

ckpt = load_state_dict(
    load_path, "model.pt", local_cache=None, map_location="cpu"
)

## CONVERT TO HF-Format
olmo_config = cfg.model
n_layers = 32
n_layers = olmo_config.n_layers
n_heads = olmo_config.n_heads
dim = olmo_config.d_model
loaded = ckpt
dims_per_head = dim // n_heads
base = 10000.0
inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))


if olmo_config.n_kv_heads is not None:
    num_key_value_heads = olmo_config.n_kv_heads  # for GQA / MQA
elif olmo_config.multi_query_attention:  # compatibility with other checkpoints
    num_key_value_heads = 1
else:
    num_key_value_heads = n_heads

dims_per_head = dim // n_heads
state_dict = {}
for layer_i in range(n_layers):
    filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
    # Unsharded
    # TODO: Layernorm stuff
    # TODO: multi query attention
    fused_dims = [dim, dims_per_head * num_key_value_heads, dims_per_head * num_key_value_heads]
    q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
        loaded[f"transformer.blocks.{layer_i}.att_proj.weight"], fused_dims, dim=0
    )
    up_proj_weight, gate_proj_weight = torch.chunk(
        loaded[f"transformer.blocks.{layer_i}.ff_proj.weight"], 2, dim=0
    )
    state_dict.update({
        f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj_weight,
        f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj_weight,
        f"model.layers.{layer_i}.self_attn.v_proj.weight": v_proj_weight,
        f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
            f"transformer.blocks.{layer_i}.attn_out.weight"
        ],
        f"model.layers.{layer_i}.mlp.gate_proj.weight": gate_proj_weight,
        f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"transformer.blocks.{layer_i}.ff_out.weight"],
        f"model.layers.{layer_i}.mlp.up_proj.weight": up_proj_weight,
    })

    # state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

state_dict.update({
    "model.embed_tokens.weight": loaded["transformer.wte.weight"],
    "lm_head.weight": loaded["transformer.ff_out.weight"]
    if "transformer.ff_out.weight" in loaded
    else loaded["transformer.wte.weight"],
})
### End of Conversion.

# Load Model.
model.load_state_dict(state_dict)
model.eval()
model.to(cfg.model.init_device)

evals = build_evaluators(cfg, device=device)

# import pdb
# pdb.set_trace()

# eval
from itertools import islice
from typing import Optional, TypeVar

T = TypeVar("T")

def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss

def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o

def get_labels(batch) -> torch.Tensor:
    # Labels are just input IDs shifted to the left (first item is ignored).
    labels, label_mask, attention_mask, instance_mask = (
        batch["input_ids"].clone(),
        batch.get("label_mask"),
        batch.get("attention_mask"),
        batch.get("instance_mask"),
    )
    if label_mask is not None:
        labels.masked_fill_(~label_mask, -100)
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0.0, -100)
    if instance_mask is not None:
        labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
    return labels[..., 1:].contiguous()

def model_forward(
        batch, loss_reduction: str = "mean", compute_z_loss: bool = False
    ):
        # shape: (batch_size, seq_len, vocab_size)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ).logits
        # import pdb
        # pdb.set_trace()
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = cross_entropy_loss(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits

def eval_batch(batch):
    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        ce_loss, _, logits = model_forward(batch, loss_reduction="none")
    return ce_loss.mean(dim=-1), logits

def eval_step(batch, evaluator):
    batch = move_to_device(batch, device)

    # Run forward pass.
    with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
        ce_loss, logits = eval_batch(batch)

    # import pdb
    # pdb.set_trace()

    # Update metrics.
    evaluator.update_metrics(
        batch, ce_loss, logits
    )  # batch includes all keys that the downstream evaluation needs
    
    return ce_loss

def to_list_all(x):
    new_d = {}
    for k, v in x.items():
        try: 
            v = v.detach().cpu().numpy().tolist()
        except:
            pass
        new_d[k] = v
    return new_d

eval_metrics = {}
# import pdb
# pdb.set_trace()
for evaluator in evals:
    # log.info(f"Running evaluation for '{evaluator.label}'...")

    # Reset metrics.
    evaluator.reset_metrics()

    # Initialize data loader iterator.
    eval_batches = iter(evaluator.eval_loader)

    # Adjust how many batches to evaluate on.
    num_eval_batches = (
        evaluator.subset_num_batches
        if evaluator.subset_num_batches is not None
        else 10000 # previously 10000
    )
    if num_eval_batches > 0:
        num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
        eval_batches = islice(eval_batches, num_eval_batches)

    # Run model over batches.
    for e_step, e_batch in tqdm(enumerate(eval_batches)):
        ce_loss = eval_step(e_batch, evaluator)

        # Get final metrics.
        metrics = evaluator.compute_metrics()
        eval_metrics.update(metrics)
        # print(eval_metrics)
        
        new_batch = to_list_all(e_batch)
        new_batch['loss']= ce_loss.detach().cpu().numpy().item()
        new_batch['task'] = evaluator.label

        with jsonlines.open(f"/home/hyeonbin/iterpre/OLMo/olmo/exp/0722_eval_all_{ckpt_num}.jsonl", "a") as writer:
            writer.write(new_batch)
    
with open(f"/home/hyeonbin/iterpre/OLMo/olmo/exp/eval_all_{ckpt_num}_summary.json", "w") as writer:
    json.dump(eval_metrics, writer)