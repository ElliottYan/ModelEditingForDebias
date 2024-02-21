import json
import argparse
import os
import numpy as np
import pandas as pd
from compute_single_metrics import compute_para_less_vs_more_all, compute_para_less_all_vs_more_all

def aggregate_item(data, unedited_results):
    locality_result = 0
    less_all_vs_more_all_total = 0
    less_vs_more_para_total = 0
    edit_success = 0
    assert len(data) == len(unedited_results)
    for id in data.keys():
        edited_metric = data[id]
        unedited_metric = unedited_results[int(id)]
        # import pdb; pdb.set_trace()
        # locality
        for locality_key in edited_metric['post']['locality'].keys():
            for ans,label in zip(edited_metric['post']['locality'][locality_key],unedited_metric[locality_key]):
                min_length = min(len(ans), len(label))
                # locality_result += np.mean(np.equal(ans, label))
                locality_result += np.mean(np.equal(ans[:min_length], label[:min_length]))
        # para
        temp = {
            "edited_lprobs": edited_metric['post']['lprobs'] ,
            "para_lprobs": edited_metric['post']['para_lprobs']
        }
        less_vs_more_para, _ = compute_para_less_vs_more_all(temp)
        less_all_vs_more_all, _ = compute_para_less_all_vs_more_all(temp)
       
        less_vs_more_para_total += less_vs_more_para
        less_all_vs_more_all_total += less_all_vs_more_all

        # edit success
        edit_success +=  edited_metric['post']["reverse_success_rate"]
    
    return {
        "edited_success": round(edit_success/len(data) *100,2),
        "locality": round(locality_result/len(data) * 100, 2),
        "less_vs_more_para": round( less_vs_more_para_total / len(data) * 100, 2),
        "less_all_vs_more_all": round(less_all_vs_more_all_total / len(data) *100, 2)
    }


                        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="FT")
    parser.add_argument('--metrics_dir', default="llama")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--only_final', action='store_true')


    args = parser.parse_args()

    metrics_dir = f"DebiasEdit/results_{args.metrics_dir}/sequential/steroset/metrics"
    # load unedited results
    unedited_file = os.path.join(metrics_dir, f"{args.alg}_pre_locality_seed{args.seed}.json")
    with open(unedited_file, "r") as f:
        unedited_data = json.load(f)
    
    # # load results
    edited_results = {}
    unedited_results = {}
    if not args.only_final:
        with open(os.path.join(metrics_dir, f"{args.alg}_extra_results_seed{args.seed}.json")) as f:
            data = json.load(f)

        for edited_number in data.keys():
            save_key = f"{args.alg}_{edited_number}"
            
            now_results = data[edited_number]
            # import pdb; pdb.set_trace()
            edited_index = len(now_results["edited"])
            edited_results[save_key] = aggregate_item(now_results["edited"], unedited_data[:edited_index])
            unedited_results[save_key] = aggregate_item(now_results["non_edited"], unedited_data[edited_index:])
        df = pd.DataFrame(unedited_results)
        df.to_csv(f"{metrics_dir}/{args.alg}_unedited_results_seed{args.seed}.csv")
        print("save to", f"{metrics_dir}/{args.alg}_unedited_results_seed{args.seed}.csv")
    # load final results
    try: 
        with open(os.path.join(metrics_dir, f"{args.alg}_results_seed{args.seed}.json")) as f:
            data = json.load(f)
        
        save_key = f"{args.alg}_All"
        edited_results[save_key] = aggregate_item(data, unedited_data)
    except:
        pass
   
    df = pd.DataFrame(edited_results)
    df.to_csv(f"{metrics_dir}/{args.alg}_edited_results_seed{args.seed}.csv")
    print("save to", f"{metrics_dir}/{args.alg}_edited_results_seed{args.seed}.csv")
    

if __name__ == "__main__":
    main()