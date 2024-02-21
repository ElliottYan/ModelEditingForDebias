ROOT=/path/to/lm-evaluation-harness-master_revise
cd $ROOT


RESULT_DIR=/path/to/DebiasEdit/results_llama/sequential/steroset/models/llama2-7b
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=/path/to/llama2-7b \
    --tasks truthfulqa_mc,winogrande,openbookqa,crows_pairs_english\
    --batch_size auto \
    --no_cache \
    --output_path  $RESULT_DIR/harness_results.json \
    --device cuda
