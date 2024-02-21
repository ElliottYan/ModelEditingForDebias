import json
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances_num_per_slice", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data")

    args = parser.parse_args()
    return args

def main(args):
    with open("data/examples_1000.json","r") as f:
        data = json.load(f)
    total_num = len(data)
    print(total_num)
    slice_num = int(total_num/args.instances_num_per_slice)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    for i in range(slice_num):
        temp_data = data[i*args.instances_num_per_slice : (i+1)*args.instances_num_per_slice]
        print(len(temp_data))
        with open(f"{args.output_dir}/examples_slice{args.instances_num_per_slice}_{i}.json", "w") as f:
            f.write(json.dumps(temp_data))
    temp_data = data[(i+1)*args.instances_num_per_slice : ]
    print(len(temp_data))
    with open(f"{args.output_dir}/examples_slice{args.instances_num_per_slice}_{i+1}.json", "w") as f:
        f.write(json.dumps(temp_data))
    

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)