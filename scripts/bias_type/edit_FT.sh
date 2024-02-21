ROOT=/path/to/DebiasEdit
cd $ROOT
RESULT_DIR=$ROOT/results_llama/sequential/steroset
mkdir -p $RESULT_DIR

# for editing_method in "FT"; do
editing_method="FT"

for bias in gender religion race profession; do
    python edit.py \
        --editing_method=$editing_method \
        --hparams_dir=our_hparams/$editing_method/llama-7b_edit_5.yaml \
        --data_dir $ROOT/../data_construction/outputs/steroset_llama/${bias}_edit.json \
        --sequential_editing \
        --metrics_save_dir $RESULT_DIR/metrics/bias_type \
        --model_save_dir $RESULT_DIR/models/$editing_method
done
