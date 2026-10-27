#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# CPU job: Phase 0 round-trip check, training pool, CPU baselines.
source "$REPO_DIR/cluster/killarney/common.sh"
activate extract

python -m llm_calc.extraction --data-dir "$RUN_ROOT/data" build-gold --n 1000 || {
    echo "Phase 0 round trip failed; see $RUN_ROOT/data/phase0_round_trip.yaml" >&2; exit 1; }
# training pool never contains a vignette from any test set
python -m llm_calc.extraction build-train --per-calculator "${TRAIN_PER_CALCULATOR:-2000}" \
    --exclude "$SETS" --out "$TRAIN" ${REVIEW_SHEET:+--review-sheet "$REVIEW_SHEET"}

for system in oracle majority template_inversion rules; do
    python -m llm_calc.extraction predict --system "$system" --sets "$SETS" --train "$TRAIN" \
        --out "$RUN_ROOT/preds"
done
