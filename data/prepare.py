import json

def read_jsonl_file(file_path):
    data = [json.loads(q) for q in open(file_path, "r")]
    return data
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res
def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Wrote json file to: {file_path}!")
    
def write_jsonl_file(file_path, data):
    with open(file_path , encoding= "utf-8",mode="w") as file: 
        for i in data: 
            file.write(json.dumps(i) + "\n")
    print(f"Wrote file at {file_path}")
    
def main():
    data = read_json_file("data/corpus/raw/fictional_knowledge_paraphrased.json")
    more_data = read_jsonl_file("data/corpus/raw/more_event_content_2023_random3670.jsonl")

    train_corpus = [{"id": ind, "text": d['train_context']} for ind, d in enumerate(data)]    
    for ind, d in enumerate(more_data):
        train_corpus.append({"id": ind, "text": d['content']})
    write_json_file("data/corpus/fictional/fictional_train_corpus.json", train_corpus)
    
    to_save = []
    for ind, da in enumerate(data):
        to_save.append({
            'type': 'original',
            'id': ind,
            'text': da['train_context'],
            'original_corpus': da['train_context'],
            'keywords':da['mem_target'],
        })
        to_save.append({
            'type': 'paraphrase',
            'id': ind,
            'paraphrase_corpus': " ".join([gen_inp + "" + gen_tar for gen_inp, gen_tar in zip(da['gen_input'], da['gen_target'])]),
            'keywords': da['gen_target']
                
        })
        for mem_inp, mem_tar, gen_inp, gen_tar in zip(da['mem_input'], da['mem_target'], da['gen_input'], da['gen_target']):
            to_save.append({
                'type': 'original_gen',
                'id': ind,
                'original_corpus': mem_inp + "" + mem_tar,
                'answer':mem_tar,                
            })            
            to_save.append({
                'type': 'paraphrase_gen',
                'id': ind,
                'paraphrase_corpus': gen_inp + "" + gen_tar,
                'answer': gen_tar
            })
    write_json_file("data/corpus/fictional/fictional_keyword.json", to_save)
            