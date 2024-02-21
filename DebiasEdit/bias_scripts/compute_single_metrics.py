import sys, json

def main():
    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf8') as f:
        jss = json.load(f)

    all_metrics = {}
    for js in jss:
        try:
            pre = js['pre']
            post = js['post']
            case_id = js['case_id']
        except:
            continue
        
        # compute pre
        # for k in ['pre', 'post']:
        for k in ['post']:
            js_ret = gather_all_metrics_each_item(js[k])
            for mkey in js_ret:
                kkey = f"{k}-{mkey}"
                # init metrics
                if kkey not in all_metrics: all_metrics[kkey] = [0, 0]
                kk_suc, kk_tot = js_ret[mkey]
                if kk_tot == 0:
                    continue
                try:
                    all_metrics[kkey][0] += kk_suc
                    all_metrics[kkey][1] += kk_tot
                except:
                    breakpoint()
    # print all results
    for key in all_metrics:
        print(f"{key}: {all_metrics[key][0]/all_metrics[key][1]*100}%")

def gather_all_metrics_each_item(js):
    ret = {}
    if "reverse_success_rate" in js:
        ret['edit_acc'] = js['reverse_success_rate'], 1
    # else:
    #     ret['edit_acc'] = None
    if 'locality' in js:
        if isinstance(js['locality']['zsre_acc'], list):
            assert len(js['locality']['zsre_acc']) == 1
            ret['loca_acc'] = js['locality']['zsre_acc'][0], 1
        else:
            ret['loca_acc'] = js['locality']['zsre_acc'], 1
    
    if 'para_lprobs' in js:
        ret['less_vs_more_para'] = compute_para_less_vs_more_all(js)
        ret['less_all_vs_more_all'] = compute_para_less_all_vs_more_all(js)
    # else:
    #     ret['less_vs_more_para'] = None
    return ret
    

def compute_para_less_vs_more_all(item):
    tot, suc = 0, 0
    try:
        more_lprob, less_lprob = item['lprobs']
    except:
        more_lprob, less_lprob = item['edited_lprobs']
    more_para_lprobs = item['para_lprobs'][0]
    for mpl in more_para_lprobs:# + [more_lprob,]:
        tot += 1
        if less_lprob > mpl:
            suc += 1
    # soft version
    # return suc, tot
    # # hard version
    if suc == tot:
        return 1, 1
    else:
        return 0, 1


def compute_para_less_all_vs_more_all(item):
    tot, suc = 0, 0
    try:
        more_lprob, less_lprob = item['lprobs']
    except:
        more_lprob, less_lprob = item['edited_lprobs']
    less_para_lprobs, more_para_lprobs = item['para_lprobs']
    
    for lpl in [less_lprob,] + less_para_lprobs:
        for mpl in [more_lprob,] + more_para_lprobs:
            if lpl > mpl:
                suc += 1
            tot += 1
    # soft version
    # return suc, tot
    # hard version
    if suc == tot:
        return 1, 1
    else:
        return 0, 1


if __name__ == "__main__":
    main()