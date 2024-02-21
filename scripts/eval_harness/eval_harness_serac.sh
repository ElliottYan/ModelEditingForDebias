ROOT=/path/to/lm-evaluation-harness-master_revise
ALG="SERAC"
template_file="/path/to/DebiasEdit/our_hparams/SERAC/llama-7b.yaml"

for seed in 0 1 2; do
    model=$ALG-seed-$seed
    RESULT_DIR=/path/to/DebiasEdit/results_llama/sequential/steroset/models/$model

    export HOME=$ROOT
    cd $ROOT

    python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/path/to/llama2-7b,serac=True,hparams_dir=$RESULT_DIR,template_dir=$template_file\
    --tasks truthfulqa_mc,winogrande,openbookqa,crows_pairs_english\
    --batch_size 1 \
    --no_cache \
    --output_path  /path/to/DebiasEdit/results_llama/sequential/steroset/${ALG}_harness_seed$seed.json \
    --device cuda
done
