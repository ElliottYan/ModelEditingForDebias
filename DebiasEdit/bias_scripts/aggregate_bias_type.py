import json
import argparse
import pandas as pd
import statistics
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="FT")
    parser.add_argument('--metrics_dir', default="DebiasEdit/results_llama/sequential/steroset/metrics/bias_type")
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    
    BIAS_TYPE = ["race", "gender","religion", "profession"]
    ALGS = ["FT", "FT_L", "MEND", "MEMIT", "ROME", "SERAC"]
    all_results = {}
    for alg in ALGS:
        for edited_bias in BIAS_TYPE:
            all_results[f"edit_{edited_bias}"] = {}
            for eval_bais in BIAS_TYPE:
                filename = f"{alg}_edit_{edited_bias}_eval_{eval_bais}.json"
                with open(os.path.join(args.metrics_dir, filename), "r")as f:
                    eval_results = json.load(f)
                success = 0
                for e in eval_results:
                    success += int(eval_results[e]["post"]["reverse_success_rate"])
                    
            
                all_results[f"edit_{edited_bias}"][f"{eval_bais}"] = round(success/len(eval_results) * 100, 2)
        
        df = pd.DataFrame(all_results)
        print(df)
        df = df.T
        if not os.path.exists(f"bias_type"):
            os.makedirs(f"bias_type")
        df.to_csv(f"bias_type/{alg}.csv")


    
    



if __name__ == "__main__":
    main()
    


        