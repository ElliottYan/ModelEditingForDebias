alg=$1
ROOT=/path/to/DebiasEdit/bias_scripts
cd $ROOT
for seed in 3 4 5 6 7 8 9 10; do
python aggregate_sequential_metrics.py \
    --seed $seed\
    --alg $alg \
    --metrics_dir llama
done
