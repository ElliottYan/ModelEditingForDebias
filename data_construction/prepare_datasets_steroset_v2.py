import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import os
from copy import deepcopy
import fire


NAN_TOKEN = [" ", ".", ""]
MODEL_NAME="/path/to/huggingface/llama2-7b"


def get_common_and_diff(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """

    seq1 = [str(x) for x in seq1.split()]
    seq2 = [str(x) for x in seq2.split()]
    
    import difflib
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    # template1, template2 = [], []
    template = []
    
    op = matcher.get_opcodes()[0]
    if op[0] == 'equal':
              
        template += [seq1[i] for i in range(op[1], op[2], 1)]
        template = " ".join(template)
        if len(seq1) > op[2] + 1 and len(seq2) > op[2] + 1:

            targets=(seq1[op[2]], seq2[op[2]])
            return template, targets
        else:
            return None, None
    else:
        return None, None
            
    

    
def run_lprobs(model, tokenizer, sentence, device, prefix=None):
    # whether it has sos or eos? GPT2 has neither...

    if prefix:
        new_sentence = prefix + sentence
    else:
        new_sentence = sentence
    
    inps = tokenizer(new_sentence, return_tensors='pt').to(device)
    target_ids = tokenizer(sentence, return_tensors='pt').to(device)
    
    input_ids = inps["input_ids"][:,:-1]
    # targets = shift_left(input_ids, pad=tokenizer.pad_token_id)
    targets_ids = target_ids["input_ids"][:,1:] #remove bos
    outputs = model(input_ids=input_ids, attention_mask=None) # [1, l, V]
    logits = outputs.logits

    lprobs = torch.nn.functional.log_softmax(logits, dim=-1) # [1, l, V]
    lprobs = lprobs[:,-targets_ids.size(1):,:]
    gather_lprobs = lprobs.gather(-1, targets_ids[:,:,None]) # [1, l, 1]
    sent_lprobs = gather_lprobs[0].sum()
    return sent_lprobs

def re_extract():
    split = "edit"
    with open(f"data_construction/outputs/v2/MEND_bias_upper_repha/{split}.json", "r")as f:
        data = json.load(f)
    
    final_results = []
    for d in data:
        temp_item = deepcopy(d)
        prompt, targets = get_common_and_diff(d["original_pairs"][0],d["original_pairs"][1])
        
        
        for i, p in enumerate(d["para_pairs"][1]):
            if d["rephrase_target"] in p:
                index = i
                break
        if len(d["para_pairs"][0]) >= index + 1 and len(d["para_pairs"][1]) >= index + 1 :
            rephrase_prompt, rephrase_targets = get_common_and_diff(d["para_pairs"][0][index], d["para_pairs"][1][index])

            if prompt is not None and targets is not None and rephrase_prompt is not None and rephrase_targets is not None:
                temp_item["prompt"] = prompt
                temp_item["target_new"] = targets
                temp_item["rephrase_prompt"] = rephrase_prompt
                temp_item["rephrase_target"] = rephrase_targets
                final_results.append(temp_item)
    with open(f"data_construction/outputs/single_target_{split}.json", "w") as f:
        json.dump(final_results, f, indent= 4)

def re_extract_v2():
    split = "edit"
    with open(f"./outputs/v2/MEND_bias_upper_repha/{split}.json", "r")as f:
        data = json.load(f)
        final_results = []

    for d in data:
        temp_item = deepcopy(d)
        prompt, targets = get_common_and_diff(d["original_pairs"][0],d["original_pairs"][1])
        subject = d["subject"].lower() 
        parts = d["original_pairs"][1].lower().split(subject)
        j = len(parts[0]+subject)
        if prompt is None or len(prompt) < len(parts[0]) + len(subject):
            assert len(parts) == 2
            prompt, targets = d["original_pairs"][1][:j], d["original_pairs"][1][j:].strip()

        # NOTE: need further attention!!!
        for i, p in enumerate(d["para_pairs"][1]):
            if d["rephrase_target"] in p:
                index = i
                break
        # if len(d["para_pairs"][0]) >= index + 1 and len(d["para_pairs"][1]) >= index + 1 :
        #     rephrase_prompt, rephrase_targets = get_common_and_diff(d["para_pairs"][0][index], d["para_pairs"][1][index])

        #     if prompt is not None and targets is not None and rephrase_prompt is not None and rephrase_targets is not None:
        temp_item["prompt"] = prompt
        temp_item["target_new"] = targets[1]
        temp_item["rephrase_prompt"] = d['rephrase_prompt']
        temp_item["rephrase_target"] = d['rephrase_target']
        final_results.append(temp_item)
                
    with open(f"./outputs/diff_subject_edit.json", "w") as f:
        json.dump(final_results, f, indent= 4)

def re_extract_postag():
    split = "edit"
    with open(f"./outputs/v2/MEND_bias_upper_repha/{split}.json", "r")as f:
        data = json.load(f)
    
    unlearn_samples = []
    learn_samples = []
    
    import spacy
    nlp = spacy.load('en_core_web_sm')
    
    for d in data:
        temp_item = deepcopy(d)
        prompt, targets = get_common_and_diff(d["original_pairs"][0],d["original_pairs"][1])
        subject = d["subject"].lower() 
        parts = d["original_pairs"][1].lower().split(subject)
        j = len(parts[0]+subject)
        if prompt is None or len(prompt) < len(parts[0]) + len(subject):
            assert len(parts) == 2
            prompt, targets = d["original_pairs"][1][:j], d["original_pairs"][1][j:].strip()
        
        # sent less
        sent_more_tok = preprocess_sentence(nlp, d['original_pairs'][0])
        sent_less_tok = preprocess_sentence(nlp, d['original_pairs'][1])

        # subject pos
        more_subj_pos = find_subject_pos(sent_more_tok, subject)
        less_subj_pos = find_subject_pos(sent_less_tok, subject)
        
        # diff pos
        diff_pos = -1
        for i in range(min(len(sent_more_tok), len(sent_less_tok))):
            if i >= len(sent_more_tok) or i >= len(sent_less_tok):
                break
            mtok = sent_more_tok[i].text
            ltok = sent_less_tok[i].text
            if mtok == ltok:
                diff_pos = i
            else:
                break
        
        try:
            assert less_subj_pos >=1 and more_subj_pos >= 1
        except:
            breakpoint()
        # pos tag filter
        more_prompt_pos = max(diff_pos, more_subj_pos)
        less_prompt_pos = max(diff_pos, less_subj_pos)
        
        more_prompts, more_targets = get_prompts_and_targets(more_prompt_pos, sent_more_tok)
        less_prompts, less_targets = get_prompts_and_targets(less_prompt_pos, sent_less_tok)
        
        ul_sample = deepcopy(d)
        ul_sample['rephrase_prompt'] = None
        ul_sample['rephrase_target'] = None
        ul_sample['prompts'] = more_prompts
        ul_sample['targets_new'] = more_targets
        ul_sample.pop('prompt')
        ul_sample.pop('target_new')
        unlearn_samples.append(ul_sample)
                
        l_sample = deepcopy(d)
        l_sample['rephrase_prompt'] = None
        l_sample['rephrase_target'] = None
        l_sample.pop('prompt')
        l_sample.pop('target_new')
        l_sample['prompts'] = less_prompts
        l_sample['targets_new'] = less_targets
    
        if len(less_targets) == 0 or len(more_targets) == 0:
            continue
        #     breakpoint()
        # for t in less_targets:
        #     if t == "":
        #         breakpoint()
        # for t in more_targets:
        #     if t == "":
        #         breakpoint()
        # if len(less_prompts) == 0 or len(more_prompts) == 0:
        #     breakpoint()
        # breakpoint()
        # if "" in l_sample['prompts'] or "" in ul_sample['prompts']:
            # breakpoint()
        learn_samples.append(l_sample)

    with open(f"./outputs/diff_subject_pos_learn.v3.json", "w") as f:
        json.dump(learn_samples, f, indent= 4)
    with open(f"./outputs/diff_subject_pos_unlearn.v3.json", "w") as f:
        json.dump(unlearn_samples, f, indent= 4)

def get_prompts_and_targets(start_pos, sent_tokens):
    mask = find_valid_pos(sent_tokens)

    prompts, targets = [], []
    start = -1
    for idx in range(start_pos, len(sent_tokens)):
        if mask[idx] == 1:
            if start == -1: start = idx
        elif start != -1:
            # append
            prompts.append("".join([tok.text_with_ws for tok in sent_tokens[:start]]))
            targets.append("".join([tok.text_with_ws for tok in sent_tokens[start:idx]]))
            # reset start
            start = -1
    if start != -1:
        prompts.append("".join([tok.text_with_ws for tok in sent_tokens[:start]]))
        targets.append("".join([tok.text_with_ws for tok in sent_tokens[start:]]))
    return prompts, targets

def detok(l):
    return ''.join(token.text_with_ws for token in l)
        
def find_valid_pos(sent_tokens):
    pos_sequence = [tok.pos_ for tok in sent_tokens]
    # valid_postags = {'NOUN', 'ADJ'}
    invalid_postags = {'DET', "AUX", "PUNCT", "PRON", "ADP"}
    ret = []
    for i, pt in enumerate(pos_sequence): 
        if pt not in invalid_postags:
           ret.append(1)
        else:
            ret.append(0)
    return ret

def find_subject_pos(sent_tok, subject):
    subject_pos = -1
    def get_str(tok):
        return "".join([item.text_with_ws for item in tok]).lower()
    
    for i in range(len(sent_tok)):
        if subject.lower() in get_str(sent_tok[:i]):
            subject_pos = i
            break
    return subject_pos
    
def preprocess_sentence(nlp, sentence):
    # Tokenize the sentence and perform POS tagging
    doc = nlp(sentence)
    # tokens = [token.text for token in doc]
    # pos_tags = [token.pos_ for token in doc]
    # tokens_with_ws = [token.text_with_ws for token in doc]

    return doc
        

def main():
    # read counterfact
    with open("data_construction/counterfact/counterfact-train.json","r") as f:
        unrelated_data = json.load(f)

    with open("data_construction/outputs/steroset/steroset.jsonl","r") as f:
        lines = f.readlines()
    model, tok = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16,
        ).to("cuda"),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token
    
    edit_samples = []
    index = 0
    for line in lines:
        
        d = json.loads(line)
        d["sent_more_para"] = d["sent_more_para"].split("\n")
        d["sent_less_para"] = d["sent_less_para"].split("\n")
        lprobs_more = run_lprobs(model, tok, d["sent_more"], device="cuda")
        lprobs_less = run_lprobs(model, tok, d["sent_less"], device="cuda")
        subject = d["subject"].lower()
        if lprobs_less < lprobs_more:
            parts = d["sent_less"].lower().split(subject)
            paraphrase = random.sample(d["sent_less_para"], 1)[0]
            paraphrase_parts = paraphrase.lower().split(subject)
        else:
            parts = d["sent_more"].lower().split(subject)
            paraphrase = random.sample(d["sent_more_para"], 1)[0]
            paraphrase_parts = paraphrase.lower().split(subject)
        print("para_parts", paraphrase_parts)
        print("parts", parts)
        
       
        if (len(parts)==2 and parts[1] not in NAN_TOKEN) and (len(paraphrase_parts)==2 and paraphrase_parts[1] not in NAN_TOKEN):
            unrelated_sample = random.sample(unrelated_data, 1)[0]
            edit_sample = {
                "case_id": index, 
                "prompt": parts[0] + subject,
                "target_new": parts[1].strip(),
                "subject": subject,
                "rephrase_prompt":paraphrase_parts[0]+subject,
                "rephrase_target":paraphrase_parts[1].strip(),
                "locality_prompt":unrelated_sample["locality_prompt"],
                "locality_ground_truth":unrelated_sample["locality_ground_truth"],
                "original_lprobs":[lprobs_more.item(), lprobs_less.item()],
                "bias_type":d["bias_type"],
                "original_pairs":[d["sent_more"],d["sent_less"]],
                "para_pairs":[d["sent_more_para"], d["sent_less_para"]]
            }
            index += 1
            edit_samples.append(edit_sample)
            unrelated_data.remove(unrelated_sample)
        
    dir = "data_construction/outputs/steroset/"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    with open(os.path.join(dir, "edit_reverse.json"), "w") as f:
        f.write(json.dumps(edit_samples))

    # MEND
    random.shuffle(edit_samples)
    dir = "data_construction/outputs/steroset/MEND"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    with open(os.path.join(dir, "train.json"), "w") as f:
        f.write(json.dumps(edit_samples[: int(0.5*len(edit_samples))]))
    with open(os.path.join(dir, "edit.json"), "w") as f:
        f.write(json.dumps(edit_samples[int(0.5*len(edit_samples))+1:int(0.9*len(edit_samples))]))
    with open(os.path.join(dir, "val.json"), "w") as f:
        f.write(json.dumps(edit_samples[int(0.9*len(edit_samples))+1:]))

    return

if __name__ == '__main__':
    # main()
    # re_extract()
    # re_extract_v2()
    fire.Fire()