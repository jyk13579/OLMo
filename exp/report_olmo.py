import wandb
# Initialize the wandb API
api = wandb.Api()

# Specify your project and entity (user/team)
entity = "wldus2756" 
project = "OLMO_PT" 
run_id = "fjf0sq45"
run = api.run(f"{entity}/{project}/{run_id}")

# 557k, 2048 1e-4, wldus2756/OLMO_PT/fjf0sq45
# 278ㅏ qnv125or

# Fetch all runs from a specific project
# runs = api.runs(f"{entity}/{project}")

# Fetch the run history as a DataFrame
history_df = run.history()
# Convert the DataFrame to a dictionary
history_dict = history_df.to_dict(orient='list')

EVALS = [
    'eval/downstream/copa_acc',
    'eval/downstream/piqa_len_norm',
    'eval/downstream/rte_len_norm',
    'eval/downstream/openbook_qa_len_norm',
    'eval/downstream/commitment_bank_acc',
    'eval/downstream/arc_easy_acc',
    'eval/downstream/hellaswag_len_norm',
    'eval/downstream/winogrande_acc',
    'eval/downstream/sst2_acc',
    'eval/downstream/mrpc_f1',
    
    'eval/forgetting/CrossEntropyLoss',
    'eval/forgetting/Perplexity',
    
    'eval/once_composition/CrossEntropyLoss', 
    'eval/once_memorization/CrossEntropyLoss', 
    'eval/once_paragraph/CrossEntropyLoss', 
    'eval/once_semantic/CrossEntropyLoss', 
    
    'eval/once_composition/Perplexity', 
    'eval/once_memorization/Perplexity', 
    'eval/once_paragraph/Perplexity', 
    'eval/once_semantic/Perplexity', 

    'eval/paraphrase_composition/CrossEntropyLoss', 
    'eval/paraphrase_memorization/CrossEntropyLoss', 
    'eval/paraphrase_paragraph/CrossEntropyLoss', 
    'eval/paraphrase_semantic/CrossEntropyLoss', 

    'eval/paraphrase_composition/Perplexity', 
    'eval/paraphrase_memorization/Perplexity', 
    'eval/paraphrase_paragraph/Perplexity', 
    'eval/paraphrase_semantic/Perplexity'
]

import math
# import pdb; pdb.set_trace()
for key in EVALS:
    data = history_dict[key]
    filtered_data = [x for x in data if not math.isnan(x)]
    tmp = [str(x) for x in filtered_data]
    print(key, "|", "|".join(tmp))
    
