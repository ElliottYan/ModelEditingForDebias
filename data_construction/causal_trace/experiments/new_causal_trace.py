import argparse
import json
import os
import re
from collections import defaultdict
import copy
import numpy
from copy import deepcopy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
import sys
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="/path/to/llama2-7b"
    )
    aa("--fact_dir", default="data_construction/outputs/steroset_llama/")
    aa("--name", default="edit")
    aa("--max-expect", default=None)
    aa("--output_dir", default="")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    args = parser.parse_args()

    # dont hard code env variables inside python codes!
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"{args.name}_n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    # torch_dtype = torch.float16 if "20b" in args.model_name else None
    torch_dtype = torch.float16 if 'llama' not in args.model_name else torch.bfloat16
    print(torch_dtype)

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    filename = os.path.join(args.fact_dir, f"{args.name}.json")
    with open(filename) as f:
        knowns = json.load(f)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])
    results = []
    for knowledge in tqdm(knowns):
        known_id = knowledge["case_id"]
        
    
        filename = f"{result_dir}/knowledge_{known_id}.npz"
        prompts, target_new = calculate_differences(mt, knowledge["prompt"], knowledge["subject"], knowledge["target_new"], noise=noise_level)
        new_item = deepcopy(knowledge)
        new_item["prompts"] = prompts
        new_item["targets_new"] = target_new
        results.append(new_item)
        print(new_item)

    with open(f"{args.output_dir}/mediate_results_{args.name}.json", "w") as f:
        json.dump(results, f ,indent=4)


def calculate_differences(mt,
    prompt,
    subject,
    target,
    noise,
    samples=2):
    # original

    inp = make_inputs(mt.tokenizer, [prompt] * samples)
    target = " "+target
    print("prompt:", prompt, "\nsubject:", subject, "\ntarget: ", target)

    assert subject in prompt
    # Our changes
    original_lprobs = predict_tokens_new(prompt, mt, target)
    print(original_lprobs)
    # Combine multiple tokens    
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    # corrupt all subject tokens
    low_lprobs = trace_with_patch_new(
        mt, inp, [], target , e_range, noise=noise
    )
    diff = original_lprobs - low_lprobs
    expected_tensor = mt.tokenizer(target, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to("cuda")
    
    _, index = torch.topk(diff, k=min(5,expected_tensor.shape[1]))
    prompts = []
    target_new = []

    for i in list(index):
        new_prompt = torch.concat((inp["input_ids"][0], expected_tensor[0,:i]), dim=-1)
        p = mt.tokenizer.decode(new_prompt, skip_special_token=True)
        t = mt.tokenizer.decode(expected_tensor[0, i])
        if t != '' and t != '.':
            prompts.append(p)
            target_new.append(t)
    
    return prompts, target_new


def trace_with_patch_new(
    mt,  # The model and tokenizer
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    target,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    # Backup
    model = mt.model
            
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # NOTE: MODIFICATION STARTS NOW.
    
    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    
    with torch.no_grad(), nethook.TraceDict(model,[embed_layername] + list(patch_spec.keys()) + additional_layers, edit_output=patch_rep,) as td:
        lprobs = predict_tokens_new(None, mt, target, inp=inp)
    

    if trace_layers is not None:
       # import pdb; pdb.set_trace()
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return lprobs, all_traced
    
    return lprobs

@torch.no_grad()
def predict_tokens_new(prompt, mt, target, samples=2, inp=None):
    # 1. pad to a single tensor
    if mt.tokenizer.pad_token_id is None:
        mt.tokenizer.pad_token_id = mt.tokenizer.eos_token_id
    if inp is None:
        # NOTE: without any special tokens
        inp = make_inputs(mt.tokenizer, [prompt]*samples) #[1, Lp] 
    
    expected_tensor = mt.tokenizer(target, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to("cuda")
    expected_tensor = expected_tensor.repeat(samples,1) 
    input_ids = torch.concat((inp['input_ids'], expected_tensor), dim=-1)[:, :-1] #[N*K, Lp]
    # expected_tensor = expected_tensor[:,1:]

    # model forward
    output = mt.model(input_ids=input_ids).logits
    # lprobs = F.log_softmax(output, dim=-1)
    probs = output[1:,-expected_tensor.shape[1]:].mean(dim=0)
    # 3. gathering
    
    expected_tensor = expected_tensor[1:,:]
    gathered_lprobs = probs.gather(-1, torch.where(expected_tensor != -100, expected_tensor, 0)).squeeze(0)
    
    return gathered_lprobs

def trace_with_patch(
    mt,  # The model and tokenizer
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answer_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    bias_ids_set=None,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
    samples=10, # Samples to average
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    # Backup
    model = mt.model
            
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                import pdb; pdb.set_trace()
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # NOTE: MODIFICATION STARTS NOW.
    
    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    
    with torch.no_grad(), nethook.TraceDict(model,[embed_layername] + list(patch_spec.keys()) + additional_layers, edit_output=patch_rep,) as td:
        lprobs = predict_tokens(None, mt, bias_ids_set, inp=inp, samples=samples)
    
    # get target_index
    
    # for i in range(bias_ids_set.shape[0]):
    #     #import pdb; pdb.set_trace()
    #     if all(bias_ids_set[i] == answer_t):
    #         target_index = i
    #         break
    
    # softmax = torch.nn.Softmax(dim=0)
    # bias_set_logsoftmax_softmax = softmax(lprobs) #[K,]
    # final_prob = bias_set_logsoftmax_softmax[target_index]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
       # import pdb; pdb.set_trace()
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return lprobs, all_traced
    
    return lprobs


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                # NOTE: even in our case, the
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
   # probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    probs = outputs_exp.logits[1:, -1, :].mean(dim=0)[answers_t]
    return probs

@torch.no_grad()
def predict_tokens(prompt, mt, expected_set, samples=10, inp=None):
    
    K = expected_set.shape[0]
    N = samples + 1
    # NOTE: The first sample is used to restore clean results!!!
    
    # 1. pad to a single tensor
    if mt.tokenizer.pad_token_id is None:
        mt.tokenizer.pad_token_id = mt.tokenizer.eos_token_id
    if inp is None:
        # NOTE: without any special tokens
        inp = make_inputs(mt.tokenizer, [prompt] * (K * N)) #[N*K, Lp] 
    #expected_tensor = pad_sequence(expected_set, batch_first=True, padding_value=mt.tokenizer.pad_token_id)[None] # [1, K, Lt]
    expected_tensor = expected_set.repeat(N,1) #[N*K, Lw]
    input_ids = inp['input_ids'] #[N*K, Lp]
    
    new_input_ids = torch.cat([input_ids, torch.zeros_like(expected_tensor).fill_(mt.tokenizer.pad_token_id)], dim = 1) # [N*K, Lp+Lt]

    for i in range(N*K):
        cur_len = (new_input_ids[i] != mt.tokenizer.pad_token_id).sum()
        new_input_ids[i, cur_len:cur_len+expected_tensor.shape[1]] = expected_tensor[i] # [N*K, Lp+Lt]

    
    # orignial_input_ids = torch.cat([inp['input_ids'][0].unsqueeze(0), torch.full((1, expected_tensor.shape[1]), mt.tokenizer.pad_token_id).to("cuda")], dim=-1) #[1, Lp+Lw]
    # all_input_ids = torch.cat([orignial_input_ids, new_input_ids], dim=0) #[N*K+1, Lp+Lw]
    all_input_ids = new_input_ids
    
    # 2. create target tokens and mask
    Lp = input_ids.shape[1]
    Lw = expected_tensor.shape[1]
    target_tokens = torch.tensor(-100, device="cuda").repeat(K, Lp+Lw)
    for i in range(K):
        cur_len = inp["attention_mask"][i].sum()
        target_token = expected_tensor[i]
        target_tokens[i, cur_len-1: cur_len-1+len(target_token)] = target_token
    target_tokens[target_tokens == mt.tokenizer.pad_token_id] = -100
    target_mask = (target_tokens != -100).float()
    
    # model forward
    output = mt.model(input_ids=all_input_ids) # [N*K, Lp+Lt, V]
    logits = output.logits

    logits = logits.reshape(N, K, logits.shape[1], logits.shape[2])
    lprobs = F.log_softmax(logits, dim=-1) # [N, K, Lp+Lt, V]
    lprobs = lprobs[1:].mean(dim=0) #[K,L,V]
    # 3. gathering
    gathered_lprobs = lprobs.gather(2, torch.where(target_tokens != -100, target_tokens, 0).unsqueeze(2)).squeeze(2) # [K, Lw+Lp, 1] -> [K, Lw+Lp]
    ret = (gathered_lprobs * target_mask)
    return ret

def preprocess_list(mt, expect, unexpect, max_expect=None):
    bias_ids_set = []
    if mt.tokenizer.pad_token_id is None:
        mt.tokenizer.pad_token_id = mt.tokenizer.eos_token_id
    
    if max_expect is None:
        max_expect = len(expect)
        max_unexpect = len(unexpect)
    else:
        max_expect = min(int(max_expect), len(expect))
        max_unexpect = min(int(max_expect), len(unexpect))
        
    for i in range(max_expect):
        if "llama" != mt.model.name:
            bias_ids_set.append(" " + expect[i])
        else:
            bias_ids_set.append(expect[i])  
    for i in range(max_unexpect):
        if "llama" != mt.model.name:
            bias_ids_set.append(" " + unexpect[i])
        else:
            bias_ids_set.append(unexpect[i])
    # NOTE: make sure there's no other special tokens -> BOS, EOS
    
    bias_ids_set = mt.tokenizer(bias_ids_set, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to("cuda")#make_inputs(mt.tokenizer, expect)['input_ids']
    
    return bias_ids_set

def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    answer_id=None,
    base_score=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    # print(bias_ids_set)
    inp = make_inputs(mt.tokenizer, [prompt] * ((samples+1)*bias_ids_set.shape[0]))
    # Our changes
    bias_difference=None
    if kind is None:
        # move all logic into predict_tokens.
        all_probs = predict_tokens(prompt, mt, bias_ids_set, samples=samples)
        
        # logsoftmax + softmax = softmax; 
        bias_set_logsoftmax_softmax = F.softmax(all_probs, dim=0)
        base_score, target_indice = torch.max(bias_set_logsoftmax_softmax,0)
        answer_t = bias_ids_set[target_indice]
        
    else:
        answer_t = torch.tensor(answer_id).to("cuda")

    # Combine multiple tokens
    answer = "".join(decode_tokens(mt.tokenizer, answer_t))
    print("\nanswer", answer, "\nanswer_t", answer_t)
    
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    
    # corrupt all subject tokens
    # low_all_probs [K,]
    low_score = trace_with_patch(
        mt, inp, [], answer_t, e_range, bias_ids_set=bias_ids_set, noise=noise, uniform_noise=uniform_noise, samples=samples
    )
    
    if not kind:
        # restore_all_probs [L, 48, K]
        differences = trace_important_states(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            bias_ids_set=bias_ids_set,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
            samples=samples
        )
    else:
        differences = trace_important_window(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            bias_ids_set=bias_ids_set,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
            samples=samples
        )
    differences = differences.detach().cpu()
    
    
    return dict(
        scores=differences,
        bias_difference=bias_difference,
        low_score=low_score.item(),
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer_id=answer_t,
        answer_text=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
        bias_set_probs=bias_set_logsoftmax_softmax if kind==None else None
    )


def trace_important_states(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    bias_ids_set=None,
    samples=10,
):
    ntoks = inp["input_ids"].shape[1]
    probs = []
    
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        prob = []
        for layer in range(num_layers):
            
            p = trace_with_patch(
                mt,
                inp,
                [(tnum, layername(mt.model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                bias_ids_set=bias_ids_set,
                samples=samples
            )
            prob.append(p) 
        
        probs.append(torch.stack(prob)) #[48, K]
    return torch.stack(probs)#[L, 48, K]

def trace_important_window(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
    bias_ids_set=None,
    samples=10,
):
    ntoks = inp["input_ids"].shape[1]
    probs = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        prob = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(mt.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            p = trace_with_patch(
                mt,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                bias_ids_set=bias_ids_set,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
                samples=samples,
            )
            
            prob.append(p)
        probs.append(torch.stack(prob))
    return torch.stack(probs)




class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        # if "llama" in model_name:
        #     model = LlamaForCausalLM.from_pretrained(model_name,  torch_dtype=torch_dtype)
        #     model.name = "llama"
        #     tokenizer = LlamaTokenizer.from_pretrained(model_name,  torch_dtype=torch_dtype)
        #     tokenizer.pad_token_id = tokenizer.eos_token_id
        # else:
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
            model.name = model_name
        nethook.set_requires_grad(False, model)
        model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model

        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    #import pdb;pdb.set_trace()
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model"):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=5, # change to 5, compute faster
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
        
    )
    plot_trace_heatmap(result, savepdf)


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    #import pdb; pdb.set_trace()
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer_text"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    # with plt.rc_context(rc={"font.family": "Times New Roman"}):
    with plt.rc_context(rc={"font.family": "DeJavu Serif"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array if t != tokenizer.pad_token_id]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    # print("tok", toks, "substring", substring)
    substring = substring.replace(" ", "")
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level

def shift_left(x, pad=1):
    """
    Shifts a tensor left by one step.

    Args:
        x: A PyTorch tensor of shape (batch_size, sequence_length, hidden_size).

    Returns:
        A PyTorch tensor of the same shape as x.
    """
    # Create a zero tensor with the same shape as x
    shifted_x = torch.zeros_like(x).fill_(pad)

    # Shift each row of the tensor left by one step
    shifted_x[:, :-1] = x[:, 1:]

    return shifted_x

def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cpu")
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
    # And this will give us variance is (degree / degree - 2) * cov.
    # Therefore if we want to match the sample variance, we should
    # reduce cov by a factor of (degree - 2) / degree.
    # In other words we should be sampling sqrt(degree - 2 / u) * sample.
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student


if __name__ == "__main__":
    main()
