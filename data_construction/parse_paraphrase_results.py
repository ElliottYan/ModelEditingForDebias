import json
import sys
import random
import re
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

def detokenize(tokens):
    detokenizer = TreebankWordDetokenizer()
    sentence = detokenizer.detokenize(tokens)
    return sentence

def tokenize(sentence):
    tokens = word_tokenize(sentence)
    return tokens

def process_paras(s):
    splits = s.split('\n')
    if len(splits) != 3:
        return []
    return splits

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
    subjects = []
    flag = False
    target = " "
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if flag == False:
            if op[0] == 'equal':
                # template1 += [x for x in range(op[1], op[2], 1)]
                # template2 += [x for x in range(op[3], op[4], 1)]
                template += [seq1[i] for i in range(op[1], op[2], 1)]
            elif op[0] == 'replace':
                template += ["{}"]
                subjects = [" ".join([seq1[i] for i in range(op[1], op[2])]), " ".join([seq2[i] for i in range(op[3], op[4])])]
                flag = True
        else:
            if op[0] == 'equal':
                target += " ".join([seq1[i] for i in range(op[1], op[2])])
            else:
                target = ""
                break
    template = " ".join(template)

    return template, subjects, target

def count_word_in_sent(word, sent):
    word_tokens = word.lower().split()
    sent_tokens = sent.lower().split()
    lw = len(word_tokens)
    ls = len(sent_tokens)
    assert ls >= lw
    cnt = 0
    for i in range(ls-lw+1):
        match = True
        for j in range(lw):
            if sent_tokens[i+j] != word_tokens[j]:
                match = False
                break
        if match:
            cnt += 1
    return cnt

def postprocess_subject(s, clean=False):
    if s.endswith('.') or s.endswith(',') or s.endswith('?'):
        return s[:-1]
    else:
        return s

def main():
    with open(sys.argv[1], 'r', encoding='utf8') as fin:
        with open(sys.argv[2], 'w', encoding='utf8') as fout:
            lines = fin.readlines()
            invalid = 0
            for line in tqdm(lines):
                js = json.loads(line.strip())
                sent_more, sent_less = js['sent_more'], js['sent_less']
                
                tok_more, tok_less = tokenize(sent_more), tokenize(sent_less)
                tok_more_str, tok_less_str = " ".join(tok_more), " ".join(tok_less)
                
                prompt, subjects, target = get_common_and_diff(tok_more_str, tok_less_str)

                if len(subjects) != 2:
                    invalid += 1
                    continue
                subject_more, subject_less = subjects

                # filter 
                sent_more_para_sents = process_paras(js['sent_more_para'])
                
                try:
                    assert count_word_in_sent(subject_more, tok_more_str) == 1
                except:
                    invalid += 1
                    continue

                sent_more_para_toks = [tokenize(s) for s in sent_more_para_sents]
                
                sent_more_para_toks = [s for s in sent_more_para_toks if count_word_in_sent(subject_more, " ".join(s))]
                
                sent_less_para_sents = []
                for t in sent_more_para_toks:
                    sent = " ".join(t)
                    ss = sent.replace(subject_more, subject_less)
                    ts = detokenize(tokenize(ss))
                    sent_less_para_sents.append(ts)

                if len(sent_more_para_toks) == 0:
                    invalid += 1
                js['sent_more_para'] = sent_more_para_sents
                js['sent_less_para'] = sent_less_para_sents
                
                # create sent less
                js['subject_more'] = subject_more
                js['subject_less'] = subject_less

                fout.write(json.dumps(js, ensure_ascii=False) + '\n')
                
    print('Invalid Count: {}'.format(invalid))

if __name__ == '__main__':
    main()