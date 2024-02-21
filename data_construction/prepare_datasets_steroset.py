import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import os
from tqdm import tqdm
import argparse
from copy import deepcopy
NAN_TOKEN = [" ", ".", ""]


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="/path/to/model")
    parser.add_argument('--counterfact_path', type=str, default="data_construction/counterfact")
    parser.add_argument('--steroset_path', type=str, default="data_construction/steroset")
    parser.add_argument('--output_path', type=str, default="data_construction/outputs/steroset_gpt2")
    args = parser.parse_args()

    # read counterfact
    with open(f"{args.counterfact_path}/counterfact-train.json","r") as f:
        unrelated_data = json.load(f)

    with open(f"{args.steroset_path}/steroset.jsonl","r") as f:
        lines = f.readlines()
    
    MODEL_NAME = args.model
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
    for line in tqdm(lines):
        
        d = json.loads(line)
        d["sent_more_para"] = d["sent_more_para"].split("\n")
        d["sent_less_para"] = d["sent_less_para"].split("\n")
        lprobs_more = run_lprobs(model, tok, d["sent_more"], device="cuda")
        lprobs_less = run_lprobs(model, tok, d["sent_less"], device="cuda")
        

        if lprobs_less < lprobs_more:
            subject_start_idx = d["sent_less"].lower().find(d["subject"].lower())
            template_soup = []
            for p in d["sent_less_para"]:
                if len(p.lower().split(d["subject"].lower())) == 2 and p.lower().split(d["subject"].lower())[1] not in NAN_TOKEN:
                    template_soup.append(p)   
            
            
        # else:
        #     parts = d["sent_more"].lower().split(subject)
        #     paraphrase = random.sample(d["sent_more_para"], 1)[0]
        #     paraphrase_parts = paraphrase.lower().split(subject)
            
            parts = d["sent_less"].split(d["subject"])
            if  (len(parts)==2 and parts[1] not in NAN_TOKEN) and len(template_soup)>0:
                paraphrase = random.sample(template_soup, 1)[0]
                paraphrase_subject_index = paraphrase.lower().find(d["subject"].lower())

                unrelated_sample = random.sample(unrelated_data, 1)[0]
                edit_sample = {
                        "case_id": index, 
                        "prompt": d["sent_less"][:subject_start_idx+ len(d["subject"])].strip(),
                        "target_new": d["sent_less"][subject_start_idx+ len(d["subject"]):].strip(),
                        "subject": d["subject"],
                        "rephrase_prompt":paraphrase[: paraphrase_subject_index + len(d["subject"])].strip(), 
                        "rephrase_target":paraphrase[paraphrase_subject_index + len(d["subject"]) :].strip(),
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
            
    # dir = "data_construction/outputs/steroset_gpt2/"
    # if not os.path.exists(dir):
    #     os.makedirs(dir, exist_ok=True)
    
    # with open(os.path.join(dir, "edit_reverse.json"), "w") as f:
    #     json.dump(edit_samples, f, indent=4)

    # MEND
    random.shuffle(edit_samples)
    dir = args.output_path
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    with open(os.path.join(dir, "train.json"), "w") as f:
        json.dump(edit_samples[: int(0.5*len(edit_samples))], f, indent=4)
    with open(os.path.join(dir, "edit.json"), "w") as f:
        json.dump(edit_samples[int(0.5*len(edit_samples))+1:int(0.9*len(edit_samples))], f, indent=4)
    with open(os.path.join(dir, "val.json"), "w") as f:
        json.dump(edit_samples[int(0.9*len(edit_samples))+1:], f, indent=4)

    return

if __name__ == '__main__':
    main()