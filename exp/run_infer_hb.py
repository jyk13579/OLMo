### Run again
import time

import sys
sys.path.append("/mnt/nas/jiyeon/OLMo")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
from tqdm.auto import tqdm
import argparse
from torch.nn import functional as F
# import json
import fcntl

from olmo.checkpoint import load_state_dict
# from olmo.eval import build_evaluators
from olmo.config import TrainConfig
from olmo.util import clean_opt
# from olmo.model import OLMo

# Access the argument


parser = argparse.ArgumentParser(description="")
parser.add_argument('--idx', type=int, help='GPU(Device) Number')
parser.add_argument('--num_device', type=int, help='Total number of GPU(Device)', default=4)
parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
parser.add_argument("--ckpt_num", type=int, help='ckpt to run')
args = parser.parse_args()

# Access the argument
gpu_idx = args.idx
ckpt_num = args.ckpt_num
NUM_DEVICES = args.num_device
BS = args.batch_size

device = torch.device(f"cuda:{gpu_idx}")
result_path = f"/mnt/nas/jiyeon/OLMo/analysis/OLMo_C4_Infer_Result/eval_{ckpt_num}_tmp.jsonl"

yaml_path = "/mnt/nas/jiyeon/OLMo/configs/official/OLMo-7B.yaml"
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

# import pdb
# pdb.set_trace()

# eval
# from itertools import islice
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

def to_list_all(x):
    new_d = {}
    for k, v in x.items():
        try: 
            v = v.detach().cpu().numpy().tolist()
        except:
            pass
        new_d[k] = v
    return new_d

def write_with_lock(result_path, result_dict):
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            with open(result_path, 'a') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with jsonlines.Writer(file) as writer:
                    writer.write(result_dict)
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            return True
        except IOError:
            attempt += 1
            time.sleep(0.1)  # Wait for 100ms before retrying
    
    return False


import torch
import torch.nn.functional as F

def get_loss_and_probabilities(model, inp):
    inp = inp.to(device)
    with torch.no_grad():
        outputs = model(inp, labels=inp)
        logits = outputs.logits
        real_loss = outputs.loss.item()

        # Compute per-token loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inp[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.size()).detach().cpu().numpy().tolist()

        # Convert logits to probabilities
        shift_probs = F.softmax(shift_logits, dim=-1)
        # Get the probabilities of the actual next tokens
        shift_labels_unsqueezed = shift_labels.unsqueeze(-1)
        token_probs = torch.gather(shift_probs, 2, shift_labels_unsqueezed).squeeze(-1).detach().cpu().numpy().tolist()

    return real_loss, per_token_loss, token_probs


# import json

from datasets import Dataset


# Load already processed files.
eval_dset = Dataset.load_from_disk("/mnt/nas/jiyeon/OLMo/data/c4_dataset/train")
# already_done = [] 

# try:
#     f = open(result_path)
#     for elem in f:
#         try:
#             already_done.append(json.loads(elem))
#         except:
#             pass
# except:
#     pass

# max_id = max([int(x['sample_idx']) for x in already_done if x['sample_idx'] % 4 == gpu_idx])
# already_done_id = 0

END_IDX = len(eval_dset)
START_IDX = 0
eval_dset = eval_dset.select(range(START_IDX, END_IDX))

# Choose only evals[0] and evals[1]. Others are downstream tasks such as PIQA, Hellaswag, etc.
for sample_idx in tqdm(range(0, END_IDX, BS)):
    
    if sample_idx % (BS * NUM_DEVICES) != gpu_idx * BS:
        continue

    batch = eval_dset[sample_idx: sample_idx + BS]

    orig_sample = batch['input_ids']
    sample = torch.LongTensor(orig_sample)
    # import pdb
    # pdb.set_trace()
    rloss, ptloss, token_probs = get_loss_and_probabilities(model, sample)

    for idx, (p_loss, t_probs) in enumerate(zip(ptloss, token_probs)):

        result_dict = {
            "sample_idx": START_IDX + sample_idx + idx,
            "sample": orig_sample[idx],
            "per_token_loss": p_loss,
            "token_probabilities": t_probs,
        }

        write_with_lock(result_path, result_dict)