import json
import numpy as np
import pandas as pd
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data_construction/outputs/steroset_llama")
    parser.add_argument('--split', default="eval")

    args = parser.parse_args()

    with open(f"{args.data_dir}/{args.split}.json", "r") as f:
        data = json.load(f)
    all_results = {}
    prompt_lengths = []
    target_lengths = []

    split_data = {}
    for d in data:
        bias_type = d["bias_type"]
        if bias_type in all_results.keys():
            all_results[bias_type]["number"] += 1
            all_results[bias_type]["prompt_length"].append(len(d["prompt"].split()))
            all_results[bias_type]["target_length"].append(len(d["target_new"].split()))
            split_data[bias_type].append(d)
        else:
            all_results[bias_type] = {}
            all_results[bias_type]["number"] = 1
            all_results[bias_type]["prompt_length"]=[len(d["prompt"].split())]
            all_results[bias_type]["target_length"]=[len(d["target_new"].split())]
            split_data[bias_type] = [d]
        prompt_lengths.append(len(d["prompt"].split()))
        target_lengths.append(len(d["target_new"].split()))
    print("Total number: ", len(data))
    for key in all_results.keys():
        p_l = all_results[key]["prompt_length"]
        t_l = all_results[key]["target_length"]
        del all_results[key]["prompt_length"]
        del all_results[key]["target_length"]
        all_results[key]["prompt_length_mean"] = round(np.mean(p_l),2)
        all_results[key]["prompt_length_std"] = round(np.std(p_l),2)
        all_results[key]["target_length_mean"] = round(np.mean(t_l),2)
        all_results[key]["target_length_std"] = round(np.std(t_l),2)
        with open(f"{args.data_dir}/{key}_{args.split}.json","w")as f:
            json.dump(split_data[key], f, indent=4)
    all_results["all"] = {}
    all_results["all"]["number"] = len(data)
    all_results["all"]["prompt_length_mean"] = round(np.mean(prompt_lengths),2)
    all_results["all"]["prompt_length_std"] = round(np.std(prompt_lengths),2)
    all_results["all"]["target_length_mean"] = round(np.mean(target_lengths),2)
    all_results["all"]["target_length_std"] = round(np.std(target_lengths),2)
    df = pd.DataFrame(all_results)
    df = df.transpose()
    print(df)
    df.to_csv(f"{args.data_dir}/{args.split}.csv")

    return

if __name__ == "__main__":
    main()