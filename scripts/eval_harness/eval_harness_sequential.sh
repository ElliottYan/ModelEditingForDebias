ROOT=/path/to/lm-evaluation-harness-master_revise
BASE_PATH=/path/to/llama-2-7b-hf
for alg in FT FT_L MEND MEMIT ROME; do
    for seed in 0 1 2; do
        model=$alg-seed-$seed
        RESULT_DIR=/path/to/DebiasEdit/result_llama/sequential/steroset/models/$model

        cp $BASE_PATH/special_tokens_map.json $RESULT_DIR
        cp $BASE_PATH/tokenizer.model $RESULT_DIR
        cp $BASE_PATH/tokenizer_config.json $RESULT_DIR
        cp $BASE_PATH//tokenizer.json $RESULT_DIR

        export HOME=$ROOT
            
        cd $ROOT

        LOG_DIR=/path/to/DebiasEdit/harness_eval_logs
        mkdir -p $LOG_DIR

        python main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$RESULT_DIR\
            --tasks truthfulqa_mc,winogrande,openbookqa,crows_pairs_english\
            --batch_size auto \
            --no_cache \
            --output_path  /path/to/DebiasEdit/results_llama/sequential/steroset/${alg}_harness_seed$seed.json \
            --device cuda |& tee $LOG_DIR/$model.log
    done
done
