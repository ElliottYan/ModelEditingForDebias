ROOT=/path/to/DebiasEdit
cd $ROOT
RESULT_DIR=$ROOT/results_llama/single/steroset/
mkdir -p $RESULT_DIR

editing_method="ROME"
python edit.py \
    --editing_method=$editing_method \
    --hparams_dir=our_hparams/$editing_method/llama-7b.yaml \
    --data_dir $ROOT/../data_construction/outputs/steroset_llama/edit.json \
    --metrics_save_dir $RESULT_DIR/metrics \
    --model_save_dir $RESULT_DIR/models/$editing_method