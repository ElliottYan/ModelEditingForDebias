import openai
import os
import time, json, random
from tqdm import tqdm
import os
import openai
import os
import pandas as pd
import copy
from openai_utils import gpt_fun
from parse_paraphrase_results import get_common_and_diff, count_word_in_sent


# para_dir = "./parallel_data/"
# data_file = 
from datasets import load_dataset
dataset = load_dataset("crows_pairs")

result_dir = "./results/"
os.makedirs(result_dir, exist_ok=True)

output_file = f"{result_dir}/crows_pairs_with_para.v2.jsonl"

start_num = 0
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    start_num = len(lines)

cnt = 0
with open(output_file, 'a', encoding='utf8') as f:
    for i in tqdm(range(start_num, len(dataset['test']))):
        item = dataset['test'][i]
        cnt += 1
        new_item = copy.deepcopy(item)
        sent_more = item['sent_more']
        sent_less = item['sent_less']
        
        prompt, subjects, target = get_common_and_diff(sent_more, sent_less)
        
        if len(subjects) != 2 or count_word_in_sent(subjects[0], sent_more) > 1:
            continue
        
        messages = [{"role":"system","content":"Can you help me paraphrase the following sentence. Please give me three candidate paraphrases, and put each paraphrase in one line. Make sure to keep the word {}".format(subjects[0])},{"role":"user","content":sent_more}]
        
        sent_more_paraphrase = gpt_fun(messages)
        # sent_less_paraphrase = gpt_fun(sent_less)
        new_item['sent_more_para'] = sent_more_paraphrase
        # new_item['sent_less_para'] = sent_less_paraphrase
        f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

# for item in dataset:
    
    # assert fn.endswith('.xlsx')
    # base_fn = fn.split('.')[0]
    # out_path = f"{result_dir}/{base_fn}.txt"
    # abs_fn = os.path.join(para_dir, fn)
    # if os.path.exists(out_path):
    #     print(f'skip {abs_fn}')
    #     continue
    # df = pd.read_excel(abs_fn)
    # prompt = "\n".join(df.iloc[:, 0].tolist())
    # out = gpt_fun(prompt, "gpt-4-32k")
    # with open(out_path, 'w', encoding='utf8') as fo:
    #     fo.write(out)
    # print(out)