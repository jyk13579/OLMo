from openai_concurrent import OpenAIChatCompletionConcurrent
from collections import defaultdict
from transformers import AutoTokenizer
import openai
from time import time
from tqdm import tqdm
import json
import random
import ast
import argparse
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    print(f"Read from{file_path}")
    return res
def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Wrote json file to: {file_path}!")
def read_jsonl_file(file_path):
    data = [json.loads(q) for q in open(file_path, "r")]
    return data

def call(args): 
    save_name = args.save_name
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    openai.api_key = api_key

    start_time = time()  # Store the start time
    print("start API call", start_time)
    requests = build_requests(args.data_type)
    openai_concurrent = OpenAIChatCompletionConcurrent(api_keys=[api_key], requests_per_minute=240, expected_response_seconds=5)
    futures = openai_concurrent.create_many(requests, streaming=True)

    responses = []
    fails = []

    gen_json_file = os.path.join(f"{save_name}.jsonl")
    usage_file = os.path.join(f"{save_name}_usage.jsonl")
    failure_file = os.path.join(f"{save_name}_gpt_failure.jsonl")
        
    with open(gen_json_file, 'a') as output_file:
        for future in tqdm(futures, total=len(requests)):
            response, success = future.result()
            
            if success:
                try:
                    responses.append(response)
                    result_json = {}
                    result_json['id'] = response['metadata']['id'] 
                    result_json['original'] = response['request']['messages'][1]['content'].strip()
                    result_json['output'] = response['response'].choices[0].message.content.strip()
                    result_json['prompt_tokens'] = response['response'].usage.prompt_tokens
                    result_json['completion_tokens'] = response['response'].usage.completion_tokens
                    result_json['total_tokens'] = response['response'].usage.total_tokens

                    output_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                    output_file.flush() 
                    
                except Exception as e:
                    # Handle potential issues, e.g., log them or print them.
                    print(f"Error processing response: {e}")
            
                if len(responses) % 100 == 0:
                    print("processing:", len(responses), "done")
            else:
                fails.append(response)

    total_tokens = [response['response'].usage.total_tokens for response in responses]
    prompt_tokens = [response['response'].usage.prompt_tokens for response in responses]

    if len(fails)>0: 
        print(f"{len(fails)} instances failed")
        write_json_file(failure_file, fails)

    input_rate = 0.15/1000000
    output_rate = 0.6/1000000
    
    with open(usage_file, 'a') as f:
        token = {}
        total = sum(total_tokens)
        prompt = sum(prompt_tokens)
        token['total_tokens'] = total
        token['prompt_tokens'] = prompt
        token['completion_tokens'] = total - prompt
        token['rate'] = prompt*input_rate + (total - prompt)*output_rate
        print(f"expected rate: {token['rate']}")
        f.write(json.dumps(token, indent=4)+'\n')
        
    print("total_token:", sum(total_tokens))
    print("prompt_tokens:", sum(prompt_tokens))

    end_time = time()  # Capture the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f'The process took {elapsed_time/60} minutes.')

def build_requests(data_type):    
    if data_type == "pubmed":
        data = read_json_file("data/corpus/pubmed.json")[4:]
        system_prompt = KEYWORD_PROMPT
        model = "gpt-4-turbo-preview"
    elif "step" in data_type:
        data = read_json_file(f"data/dolma/{args.data_type}.json")
        system_prompt = TS_PROMPT
        model = "gpt-4o-mini"
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
                
    print(f"Running for {len(data)}")
    requests = []
    for index, item in enumerate(data):
        if data_type == "pubmed":
            user_prompt = f"original_corpus: '{item['text']}'"
        elif "step" in data_type:
            tokens = tokenizer.tokenize(item['text'])[:1024]
            user_prompt = f"input: {tokens}"
            
            
        requests.append(
            {
                'metadata': {'id': item['id']},
                'request': {
                    "model": model,
                    "response_format":{ "type": "json_object" },
                    "messages": [
                        {
                            'role': 'system',
                            'content': system_prompt
                        },
                        {
                            'role': 'user',
                            'content': user_prompt,
                        }
                    ],
                    "n": 1,
                    "temperature": 1,
                    "max_tokens" : 16383,
                },
            }
        )
    return requests

KEYWORD_PROMPT = "You are a helpful assistant designed to output JSON. Please 1) paraphrase the provided corpus and 2) select a few words included in the original corpus as well as paraphrase corpus verbatim. The words should be named entity or specific jargon, crucial for understanding the scientific content in the corpus. \n\n [Example]\n'original_corpus':'Antibodies were prepared in sheep against purified plasma membranes from pig adipocytes. Western (immuno) blotting revealed reactions of the antisera with a large number of proteins in adipocyte plasma membranes but remarkably few in plasma membranes from muscle, kidney, liver, lung, brain, spleen, and erythrocytes. This illustrated the high degree of specificity the serum had for adipose tissue. When injected into localized subcutaneous sites such antisera were able to cause considerable adipocyte destruction, which resulted in complete loss of adipose tissue from the site for > or = 14 wk. This cell destruction was probably mediated in part by lymphocytic infiltration. Subcutaneous injections were of limited use because of the localized nature of the effects, but, when treatment was administered intraperitoneally, systemic effects were produced that resulted in a 30% reduction in backfat thickness in the region of the last rib and a 25% reduction in fat content of fore- and hind-loin joints that resulted in a significant increase in the percentage of lean tissue. Total feed intake, live weight gain, hot carcass weights, and dressing percentage were unaffected. These results demonstrate the potential for producing long-term reductions in body fat in pigs by an immunization technique that may also provide the unexpected, potential benefit of increased lean deposition. This suggests that fat deposition per se exerts a restrictive influence on lean carcass development.' \n response: {'paraphrase_corpus': 'Sheep were immunized to create antibodies targeting the purified plasma membranes from pig adipocytes. Western blot analysis showed that these antibodies reacted with numerous proteins in the adipocyte plasma membranes, yet displayed minimal interaction with proteins from plasma membranes of muscle, kidney, liver, lung, brain, spleen, and erythrocytes. This indicated a high specificity of the serum for adipose tissue. When these antibodies were administered into specific subcutaneous areas, they led to significant adipocyte destruction, resulting in the total disappearance of fat at the injection site for at least 14 weeks. The destruction of these cells was likely aided by lymphocytic infiltration. Subcutaneous administration showed limited utility due to its localized impact, but systemic effects were observed when the treatment was given intraperitoneally. This systemic application caused a 30% decrease in backfat thickness near the last rib and a 25% reduction in fat content in the fore- and hind-loin cuts, leading to a notable rise in lean tissue proportion. Despite these changes in body composition, total feed consumption, live weight gain, hot carcass weights, and dressing percentages remained stable. These findings highlight the capability of this immunization approach to achieve sustained body fat reduction in pigs and suggest a possible additional benefit of enhancing lean tissue accumulation, indicating that fat layers may restrict lean tissue growth in carcasses.', 'named_entities': ['Antibodies', 'adipocyte plasma membranes', 'adipose tissue', 'lymphocytic infiltration', 'intraperitoneally', 'backfat thickness', 'lean tissue']} \n[End of Example]\n"

TS_PROMPT = "You are a helpful assistant designed to output JSON. Please classify each token according to its role in the context: Named Entity(0), semantic token(1) or syntactic token(2). Please keep the order of the tokens. \n\n [example] \n input : ['Apple', 'Ġis', 'Ġlooking', 'Ġat', 'Ġbuying', 'ĠU', '.', 'K', '.', 'Ġstar', 'up', 'Ġfor', 'Ġ$', '1', 'Ġbillion'] \n response : [('Apple', 0), ('Ġis', 2), ('Ġlooking', 1), ('Ġat', 2), ('Ġbuying', 1), ('ĠU', 0), ('.', 2), ('K', 0), ('.', 2), ('Ġstar', 1), ('up', 1), ('Ġfor', 2), ('Ġ$', 1), ('1', 1), 'Ġbillion', 1)] \n[end of example]"
# [example]
# input : ['Apple', 'Ġis', 'Ġlooking', 'Ġat', 'Ġbuying', 'ĠU', '.', 'K', '.', 'Ġstar', 'up', 'Ġfor', 'Ġ$', '1', 'Ġbillion']
# response : 
# ['Apple' : 0, 'Ġis' : 2, 'Ġlooking' : 1, 'Ġat' : 2, 'Ġbuying' : 1, 'ĠU' : 0, '.' : 2, 'K' : 0, '.' : 2, 'Ġstar' : 1, 'up' : 1, 'Ġfor' : 2, 'Ġ$' : 1, '1' : 1, 'Ġbillion' : 1]
# [('Apple', 0), ('Ġis', 2), ('Ġlooking', 1), ('Ġat', 2), ('Ġbuying', 1), ('ĠU', 0), ('.', 2), ('K', 0), ('.', 2), ('Ġstar', 1), ('up', 1), ('Ġfor', 2), ('Ġ$', 1), ('1', 1), ('Ġbillion', 1)]

def process_after_call():
    gpt = read_jsonl_file("data/gpt/pubmed_slot_0.jsonl") + read_jsonl_file("data/gpt/pubmed_slot_1.jsonl")
    sample = read_json_file("data/gpt/sample.json")
    
    to_save_new = []
    wrong_gpt = []
    no_keyword = []
    for ind, item in enumerate(gpt):
        original = item['original'].replace("original_corpus:", "").strip().strip("'")
        
        try:
            gpt_result = ast.literal_eval(item['output'])
            para = gpt_result['paraphrase_corpus']
            ne = gpt_result['named_entities']
        except Exception as e:
            print(f"Error processing string: {e}, {item} ")
            wrong_gpt.append(item)
            continue
        
        keywords = [word for word in ne if " "+word in original and " "+word in para]
        
        if len(keywords) == 0:
            no_keyword.append({
                "id": item['id'],
                "original_corpus": original,
                "paraphrase_corpus": para,
                "keywords": ne,
            })
        else:
            keyword_gen = [ k for k in keywords if len(k.split())==1]
            if len(keyword_gen)>0:
                keyword_gen = random.sample(keyword_gen, 1)
            else:
                keyword_gen = random.sample(keywords,1)
            
            instance = {
                "id": item['id'],
                "original_corpus": original,
                "paraphrase_corpus": para,
                "keywords": keywords,
                "answer": keyword_gen[0]
            }
            to_save_new.append(instance)
    to_save_new += sample
    # import pdb; pdb.set_trace()
    write_json_file("data/corpus/pubmed_keyword.json", to_save_new)      
    write_json_file("data/gpt/no_keyword.json", no_keyword)                 
        
def postprocess_token_classification(step):
    gpt = read_jsonl_file(f"data/gpt/token_classification_step{step}next1k_0.jsonl")
    dolma = read_json_file(f"data/dolma/step_{step}_next_1k.json")
    import ast
    to_save_new = {}
    wrong_gpt = []
    for ind, item in enumerate(gpt):
        original = item['original'].replace("input:", "").strip()
        original = ast.literal_eval(original)
        output = item['output'].replace("\n", "").strip()
        if output[-2:] == "[\"":
            output = output[:-2].strip()
        if output[-1] != "}":
            if output[-1] != "]":
                output += "]"
            output += "}"
        try:
            gpt_result = ast.literal_eval(output)
            response = gpt_result.get('response', gpt_result.get('tokens', gpt_result.get('result', None)))
            if response is not None:
                to_save_new[item['id']] = {'output': response, 'original': original}
        except Exception as e:
            wrong_gpt.append((item['output'], output))
            continue
            # print(f"Error processing string: {e}, {item} ")

    
    for passage_id, item in to_save_new.items():
        try:
            if isinstance(item['output'][0], dict):
                print(passage_id)
                temp = []
                for temp_dict in item['output']:
                    temp.append(list(temp_dict.values()))
                to_save_new[passage_id]['output']=temp
        except:
            print("$$$$$$$$$$$", passage_id)
            
    import torch
    id2idx = {item['id']:idx for idx, item in enumerate(dolma)}
    tensor = torch.full((1000, 2048), 9)
    stats = 0
    for passage_id, item in to_save_new.items():
        original, output = item['original'], item['output']
        if isinstance(output, int) or isinstance(original, int):
            print(passage_id)
            continue
        stats += len(output)    
        passage_idx = id2idx[passage_id]
        idx_original = 0
        idx_output = 0
        while idx_output < len(output) and idx_original < len(original):
            if isinstance(output[idx_output], int) or len(output[idx_output]) != 2:
                idx_output += 1
                idx_original += 1
                continue
            token, label = output[idx_output]
            original_token = original[idx_original]
            if original_token.replace("Ġ", "") == token.replace("Ġ", ""):
                if isinstance(label, int):
                    tensor[passage_idx, idx_original] = label
                idx_output += 1
                idx_original += 1
            else:
                idx_original += 1
                if idx_output+1 < len(output):
                    try:
                        if token+output[idx_output+1][0] == original_token:
                            idx_output += 2
                        elif output[idx_output+1][0] == original_token:
                            idx_output += 2
                    except:
                        idx_output += 1
        print(passage_id, len(output), sum(tensor[passage_idx] != 9).item())
    print("ALL GPT output :", stats)
    print("ALL tokens that aren't 9: ", sum(sum(tensor!=9)).item())
    print("ALL tokens that are named entity: ", sum(sum(tensor==0)).item())
    print("ALL tokens that are semantic tokens: ", sum(sum(tensor==1)).item())
    print("ALL tokens that are syntactic tokens ", sum(sum(tensor==2)).item())
    
    torch.save(tensor.detach(), f"data/dolma/token_classified/token_classification_dolma_{step}.pt")
    
# for passage_id, item in to_save_new.items():
#     original, output = item['original'], item['output']
#     passage_idx = id2idx[passage_id]
#     idx_original = 0
#     idx_output = 0
#     while idx_output < len(output) and idx_original < len(original):
#         if len(output[idx_output]) != 2:
#             idx_output += 1
#             idx_original += 1
#             continue        
#         token, label = output[idx_output]
#         original_token = original[idx_original]        
#         if original_token.replace("Ġ", "") == token.replace("Ġ", ""):
#             if isinstance(label, int):
#                 tensor[passage_idx, idx_original] = label
#             idx_output += 1
#             idx_original += 1
#         else:
#             combined_token = token
#             matched = False         
#             idx_original += 1   
#             # Try to combine with the next few tokens (up to 3 more tokens)
#             for lookahead in range(1, 4):
#                 if idx_output + lookahead < len(output):
#                     combined_token += output[idx_output + lookahead][0]
#                     if combined_token.replace("Ġ", "") == original_token.replace("Ġ", "") or output[idx_output + lookahead][0].replace("Ġ", "") == original_token.replace("Ġ", ""):
#                         idx_output += lookahead + 1
#                         matched = True
#                         break
#                 else:
#                     break          
    
# data = read_json_file("data/dolma/step_5000_prev_1k.json")
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
# full_mask = []
# for item in data:
#     attention_mask = tokenizer(item['text'], max_length=2048, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"].squeeze(0) 
#     full_mask.append(attention_mask[1:])
# full_mask = torch.stack(full_mask)
# all = torch.sum(full_mask, dim=1)
# abc = torch.load("checkpoints/pretrained/dolma_prob/dolma5000/model557000_gold_prob_dolma5000_new.pt")
# position = 0
# average_prob = torch.zeros(1000)
# for idx in range(1000):
#     length = all[idx].item()
#     prob_sum = abc[position:position+length].sum()
#     avg = prob_sum/length
#     average_prob[idx] = avg
#     position += length
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--data_type", type=str, default=None) #step_0_next_1k
    parser.add_argument("--call", type=bool, default=False)
    
    args = parser.parse_args()
    # call("data/gpt/pubmed_slot_1")
    # process_after_call()
    
    if args.call:
        call(args)
    else:
        postprocess_token_classification(args.data_type)
    
    
    """
        python data/gpt_call.py --save_name data/gpt/token_classification_step0next1k_0 --data_type step_0_next_1k
        python data/gpt_call.py --save_name data/gpt/token_classification_step5000next1k_0 --data_type step_5000_next_1k --call True
        python data/gpt_call.py --save_name data/gpt/token_classification_step110000next1k_0 --data_type step_110000_next_1k --call True
        python data/gpt_call.py --save_name data/gpt/token_classification_step278000next1k_0 --data_type step_278000_next_1k --call True
        python data/gpt_call.py --save_name data/gpt/token_classification_step432000next1k_0 --data_type step_432000_next_1k --call True
        python data/gpt_call.py --save_name data/gpt/token_classification_step556000next1k_0 --data_type step_556000_next_1k --call True
        
        python data/gpt_call.py --data_type 5000
    """