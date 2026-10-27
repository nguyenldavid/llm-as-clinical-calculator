#!/bin/bash
# GPU, time and memory come from submit.sh (models.tsv). MODEL is a Hub id.
source "$REPO_DIR/cluster/killarney/common.sh"
activate extract
name="encoder-$(basename "$MODEL")"
out="$RUN_ROOT/models/$name"
python -m llm_calc.extraction train-encoder --base-model "$MODEL" --out "$out" $(train_arg) \
    --epochs "${EPOCHS:-4}" --batch-size "${BATCH_SIZE:-16}" --lr "${LR:-3e-5}" $EXTRA
python -m llm_calc.extraction predict --system encoder --model-dir "$out" --name "$name" \
    --sets "$SETS" --out "$RUN_ROOT/preds"
