from openai_concurrent import OpenAIChatCompletionConcurrent
from collections import defaultdict
from transformers import AutoTokenizer
import openai
from time import time
from tqdm import tqdm
import json
import random
import ast

import os 
import sys
from pathlib import Path
import copy



def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res
def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Wrote json file to: {file_path}!")
def read_jsonl_file(file_path):
    data = [json.loads(q) for q in open(file_path, "r")]
    return data
def call(save_name): 
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    openai.api_key = api_key
    data = read_json_file("data/corpus/pubmed.json")[4:]
    requests = []
    print(f"Running for {len(data)}")
    for index, item in enumerate(data):
        user_prompt = f"original_corpus: '{item['text']}'"
        system_prompt = KEYWORD_PROMPT

        requests.append(
            {
                'metadata': {'id': item['id']},
                'request': {
                    "model": "gpt-4-turbo-preview",
                    # "model": "gpt-3.5-turbo",
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
                    "temperature": 0.8,
                },
            }
        )

    start_time = time()  # Store the start time
    print("start API call", start_time)

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

    with open(usage_file, 'a') as f:
        token = {}
        total = sum(total_tokens)
        prompt = sum(prompt_tokens)
        token['total_tokens'] = total
        token['prompt_tokens'] = prompt
        token['completion_tokens'] = total - prompt
        token['rate'] = (prompt*0.01 + (total - prompt)*0.03)/1000
        print(f"expected rate: {token['rate']}")
        f.write(json.dumps(token, indent=4)+'\n')
        
    print("total_token:", sum(total_tokens))
    print("prompt_tokens:", sum(prompt_tokens))

    end_time = time()  # Capture the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f'The process took {elapsed_time/60} minutes.')

KEYWORD_PROMPT = "You are a helpful assistant designed to output JSON. Please 1) paraphrase the provided corpus and 2) select a few words included in the original corpus as well as paraphrase corpus verbatim. The words should be named entity or specific jargon, crucial for understanding the scientific content in the corpus. \n\n [Example]\n'original_corpus':'Antibodies were prepared in sheep against purified plasma membranes from pig adipocytes. Western (immuno) blotting revealed reactions of the antisera with a large number of proteins in adipocyte plasma membranes but remarkably few in plasma membranes from muscle, kidney, liver, lung, brain, spleen, and erythrocytes. This illustrated the high degree of specificity the serum had for adipose tissue. When injected into localized subcutaneous sites such antisera were able to cause considerable adipocyte destruction, which resulted in complete loss of adipose tissue from the site for > or = 14 wk. This cell destruction was probably mediated in part by lymphocytic infiltration. Subcutaneous injections were of limited use because of the localized nature of the effects, but, when treatment was administered intraperitoneally, systemic effects were produced that resulted in a 30% reduction in backfat thickness in the region of the last rib and a 25% reduction in fat content of fore- and hind-loin joints that resulted in a significant increase in the percentage of lean tissue. Total feed intake, live weight gain, hot carcass weights, and dressing percentage were unaffected. These results demonstrate the potential for producing long-term reductions in body fat in pigs by an immunization technique that may also provide the unexpected, potential benefit of increased lean deposition. This suggests that fat deposition per se exerts a restrictive influence on lean carcass development.' \n response: {'paraphrase_corpus': 'Sheep were immunized to create antibodies targeting the purified plasma membranes from pig adipocytes. Western blot analysis showed that these antibodies reacted with numerous proteins in the adipocyte plasma membranes, yet displayed minimal interaction with proteins from plasma membranes of muscle, kidney, liver, lung, brain, spleen, and erythrocytes. This indicated a high specificity of the serum for adipose tissue. When these antibodies were administered into specific subcutaneous areas, they led to significant adipocyte destruction, resulting in the total disappearance of fat at the injection site for at least 14 weeks. The destruction of these cells was likely aided by lymphocytic infiltration. Subcutaneous administration showed limited utility due to its localized impact, but systemic effects were observed when the treatment was given intraperitoneally. This systemic application caused a 30% decrease in backfat thickness near the last rib and a 25% reduction in fat content in the fore- and hind-loin cuts, leading to a notable rise in lean tissue proportion. Despite these changes in body composition, total feed consumption, live weight gain, hot carcass weights, and dressing percentages remained stable. These findings highlight the capability of this immunization approach to achieve sustained body fat reduction in pigs and suggest a possible additional benefit of enhancing lean tissue accumulation, indicating that fat layers may restrict lean tissue growth in carcasses.', 'named_entities': ['Antibodies', 'adipocyte plasma membranes', 'adipose tissue', 'lymphocytic infiltration', 'intraperitoneally', 'backfat thickness', 'lean tissue']} \n[End of Example]\n"


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
                   
            
if __name__ == "__main__":
    
    # call("data/gpt/pubmed_slot_1")
    process_after_call()