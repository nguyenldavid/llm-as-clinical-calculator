#!/bin/bash
# Contamination probe for one open model: raw-text completion of T5 (public) vs T3 (fresh) notes.
source "$REPO_DIR/cluster/killarney/common.sh"
activate vllm
tp=$(echo "$EXTRA" | grep -o -- '--tensor-parallel-size [0-9]*' || true)
python -m llm_calc.extraction.contamination run --model "$(local_model "$MODEL")" \
    --out "$RUN_ROOT/contamination" $tp
python -m llm_calc.extraction.contamination summarise --dir "$RUN_ROOT/contamination"
