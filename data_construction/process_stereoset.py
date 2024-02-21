import openai
import os
import time, json, random
from tqdm import tqdm
import os
import openai
import os
import pandas as pd
import copy

from openai_utils import chatgpt_paraphrase

from datasets import load_dataset
intra_dataset = load_dataset("stereoset", "intrasentence")
inter_dataset = load_dataset("stereoset", "intersentence")
breakpoint()


for item in intra_dataset['validation']:
    new_item = {}
    sent_dict = item['sentences']
    gold_label = sent_dict['gold_label']
    for i, l in enumerate(gold_label):
        # anti-stereotype
        if l == 0:
            new_item['sent_less'] = sent_dict['sentence'][i]
        # stereotype
        elif l == 1:
            new_item['sent_more'] = sent_dict['sentence'][i]
    new_item['subject'] = item['target']

for item in inter_dataset['validation']:
    new_item = {}
    sent_dict = item['sentences']
    gold_label = sent_dict['gold_label']
    new_item['context'] = item['context']
    for i, l in enumerate(gold_label):
        # anti-stereotype
        if l == 0:
            new_item['sent_less'] = item['context'] + sent_dict['sentence'][i]
        # stereotype
        elif l == 1:
            new_item['sent_more'] = item['context'] + sent_dict['sentence'][i]
    new_item['subject'] = item['target']

result_dir = "./results/"
os.makedirs(result_dir, exist_ok=True)

# make paraphrases

output_file = f"{result_dir}/steroset.jsonl"

cnt = 0
with open(output_file, 'w', encoding='utf8') as f:
    for item in tqdm(inter_dataset['test']):
        # if cnt >= 3:
            # break
        cnt += 1
        new_item = copy.deepcopy(item)
        sent_more = item['sent_more']
        sent_less = item['sent_less']
        sent_more_paraphrase = gpt_fun(sent_more)
        sent_less_paraphrase = gpt_fun(sent_less)
        new_item['sent_more_para'] = sent_more_paraphrase
        new_item['sent_less_para'] = sent_less_paraphrase
        f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
