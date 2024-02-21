
ROOT=/path/to/DebiasEdit
cd $ROOT
RESULT_DIR=$ROOT/results_pos/sequential/steroset
mkdir -p $RESULT_DIR

# for editing_method in "FT"; do
editing_method="FT"
for seed in 3 4 5 6 7 8 9 10; do
    python edit.py \
        --editing_method=$editing_method \
        --hparams_dir=our_hparams/$editing_method/llama-7b_edit_5.yaml \
        --data_dir /path/to/data_construction/outputs/postag_llama \
        --sequential_editing \
        --seed $seed \
        --metrics_save_dir $RESULT_DIR/metrics \
        --model_save_dir $RESULT_DIR/models/$editing_method \
        --only_some_step
done
