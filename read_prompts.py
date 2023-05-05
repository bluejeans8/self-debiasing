import json

with open("realtoxicityprompts-data/prompts.jsonl",'r') as rf:
    with open("example/input_ex.jsonl", 'w') as wf:
        json_list = list(rf)

        for json_str in json_list:
            result = json.loads(json_str)
            wf.write(json.dumps(result['prompt'])+'\n')

