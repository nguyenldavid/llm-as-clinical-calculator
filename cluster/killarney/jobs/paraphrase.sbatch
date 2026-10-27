#!/bin/bash
# Candidate rewordings for clinician review (T2: MODE=options, T3: MODE=vignettes).
source "$REPO_DIR/cluster/killarney/common.sh"
activate vllm
sheet="$RUN_ROOT/data/review_${MODE}_$(basename "$MODEL").tsv"
if [ "$MODE" = options ]; then
    python -m llm_calc.extraction export-options --out "$RUN_ROOT/data/options_to_paraphrase.tsv"
    python -m llm_calc.extraction.paraphrase options --model "$MODEL" --options "$RUN_ROOT/data/options_to_paraphrase.tsv" \
        --out "$sheet" $EXTRA
else
    python -m llm_calc.extraction.paraphrase vignettes --model "$MODEL" --out "$sheet" $EXTRA
fi
echo "Review sheet: $sheet  (fill the 'approved' column, then build-t2 / build-t3)"
