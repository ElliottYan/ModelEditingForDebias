import json
import argparse
import pandas as pd
import statistics
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default="DebiasEdit/results_llama/sequential/steroset/")
    parser.add_argument('--seed_list', nargs='+')
    parser.add_argument('--alg_list', nargs='+')
    args = parser.parse_args()

    results = {}
    for alg in args.alg_list:
        results[alg] = {}
        for seed in args.seed_list:
            harness_file = os.path.join(args.model_dir, f"{alg}_harness_seed{seed}.json")
        
            with open(harness_file, "r") as f:
                data = json.load(f)

            if "crows_pairs" in results[alg].keys():
                results[alg]["crows_pairs"].append(data["results"]["crows_pairs_english"]["pct_stereotype"] * 100)
                results[alg]["openbookqa"].append(data["results"]["openbookqa"]["acc"] * 100)
                results[alg]["truthfulqa_mc"].append(data["results"]["truthfulqa_mc"]["mc2"] * 100)
                results[alg]["winogrande"].append(data["results"]["winogrande"]["acc"] * 100)
            else:
                results[alg]["crows_pairs"] = [data["results"]["crows_pairs_english"]["pct_stereotype"] *100]
                results[alg]["openbookqa"] =  [data["results"]["openbookqa"]["acc"]*100]
                results[alg]["truthfulqa_mc"] =  [data["results"]["truthfulqa_mc"]["mc2"]*100]
                results[alg]["winogrande"] =  [data["results"]["winogrande"]["acc"]*100]
        for key in results[alg].keys():
            value = results[alg][key]
            results[alg][key] = f'{statistics.mean(value):.2f} ± {statistics.stdev(value):.2f}' 
    df = pd.DataFrame(results)
    df = df.T
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    df.to_csv(os.path.join(args.model_dir, "all_harness.csv"))
    print(df)

if __name__ == "__main__":
    main()
    


        