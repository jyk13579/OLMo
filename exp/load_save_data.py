
import os
import sys
# Get the absolute path of the repository root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
print(f"root path is {root_path}")

from datasets import load_dataset
dataset = load_dataset('hbin0701/OLMo_C4_data')

print(f"Saving dataset into {root_path}/data/c4_dataset")
os.makedirs(f"{root_path}/data/c4_dataset", exist_ok=True)
dataset.save_to_disk(f"{root_path}/data/c4_dataset")
