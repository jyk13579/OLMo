
import os
import sys
from datasets import load_dataset
import argparse
# Get the absolute path of the repository root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
print(f"root path is {root_path}")

def get_c4():
    dataset = load_dataset('hbin0701/OLMo_C4_data')

    print(f"Saving dataset into {root_path}/data/c4_dataset")
    os.makedirs(f"{root_path}/data/c4_dataset", exist_ok=True)
    dataset.save_to_disk(f"{root_path}/data/c4_dataset")

def get_c4_short():
    dataset = load_dataset('jiyeonkim/c4_random240k_1k')

    print(f"Saving dataset into {root_path}/data/c4_dataset_train/c4_random240k_1k")
    os.makedirs(f"{root_path}/data/c4_dataset_train/c4_random240k_1k", exist_ok=True)
    dataset.save_to_disk(f"{root_path}/data/c4_dataset_train/c4_random240k_1k")

def get_pubmed():
    
    dataset = load_dataset('jiyeonkim/pubmed_random240k_1k')

    print(f"Saving dataset into {root_path}/data/pubmed_dataset_train/pubmed_random240k_1k")
    os.makedirs(f"{root_path}/data/pubmed_dataset_train/pubmed_random240k_1k", exist_ok=True)
    dataset.save_to_disk(f"{root_path}/data/pubmed_dataset_train/pubmed_random240k_1k")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.name == "c4":
        get_c4_short()
    elif args.name == "pubmed":
        get_pubmed()