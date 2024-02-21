alg=$1
ROOT=/path/to/DebiasEdit/bias_scripts
cd $ROOT
python aggregate_all_seeds.py \
    --metrics_dir llama \
    --model llama-7b\
    --seed_list 0 1 2 3 4 5 6 7 8 9 10 \
    --alg_list FT FT_L SERAC MEND MEMIT ROME