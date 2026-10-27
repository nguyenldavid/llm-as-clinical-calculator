# Shared body of vllm.sbatch and nuextract.sbatch (sourced, not submitted).
source "$REPO_DIR/cluster/killarney/common.sh"
activate vllm
python -m llm_calc.extraction predict --system "$KIND" --model "$MODEL" --sets "$SETS" \
    --out "$RUN_ROOT/preds" $(train_arg) --max-model-len "${MAX_MODEL_LEN:-8192}" $EXTRA
