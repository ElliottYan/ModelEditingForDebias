import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import os
NAN_TOKEN = [" ", ".", ""]
MODEL_NAME="/mnt/path/to/llama2-7b"
def run_lprobs(model, tokenizer, sentence):
    # whether it has sos or eos? GPT2 has neither...
    inps = tokenizer(sentence, return_tensors='pt').to('cuda')
    input_ids = inps.input_ids[:,:-1]
    # targets = shift_left(input_ids, pad=tokenizer.pad_token_id)
    targets = inps.input_ids[:,1:]
    logits = model(input_ids).logits # [1, l, V]
    lprobs = torch.nn.functional.log_softmax(logits, dim=-1) # [1, l, V]
    gather_lprobs = lprobs.gather(-1, targets[:,:,None]) # [1, l, 1]
    sent_lprobs = gather_lprobs[0].sum()
    return sent_lprobs

def main():
    # read counterfact
    with open("data_construction/counterfact/counterfact-train.json","r") as f:
        unrelated_data = json.load(f)

    with open("data_construction/outputs/crows_pairs_with_para.v2.parse.jsonl","r") as f:
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
        lprobs_more = run_lprobs(model, tok, d["sent_more"])
        lprobs_less = run_lprobs(model, tok, d["sent_less"])
        if lprobs_less < lprobs_more:
            subject = d["subject_less"].lower()
            parts = d["sent_less"].lower().split(subject)
            paraphrase = random.sample(d["sent_less_para"], 1)[0]
            paraphrase_parts = paraphrase.lower().split(subject)
            count = 0
            while len(paraphrase_parts) <2 and count <5:
                paraphrase = random.sample(d["sent_less_para"], 1)[0]
                paraphrase_parts = paraphrase.lower().split(subject)
                count += 1
        else:
            subject = d["subject_more"].lower()
            parts = d["sent_more"].lower().split(subject)
            paraphrase = random.sample(d["sent_more_para"], 1)[0]
            paraphrase_parts = paraphrase.lower().split(subject)
            count = 0
            while len(paraphrase_parts) <2 and count < 5:
                paraphrase = random.sample(d["sent_more_para"], 1)[0]
                paraphrase_parts = paraphrase.lower().split(subject)
                count += 1
        print("para_parts", paraphrase_parts)
        print("parts", parts)
        
       
        if (len(parts)==2 and parts[1] not in NAN_TOKEN) and (count < 5 and paraphrase_parts[1] not in NAN_TOKEN):
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
        
    dir = "data_construction/outputs/crows_pairs/"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    with open(os.path.join(dir, "edit_reverse.json"), "w") as f:
        f.write(json.dumps(edit_samples))

    # MEND
    random.shuffle(edit_samples)
    per_length = int((len(edit_samples)-10)/2)
    dir = "data_construction/outputs/crows_pairs/MEND"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    with open(os.path.join(dir, "train.json"), "w") as f:
        f.write(json.dumps(edit_samples[:per_length]))
    with open(os.path.join(dir, "edit.json"), "w") as f:
        f.write(json.dumps(edit_samples[per_length:-10]))
    with open(os.path.join(dir, "val.json"), "w") as f:
        f.write(json.dumps(edit_samples[-10:]))

    return

if __name__ == '__main__':
    main()