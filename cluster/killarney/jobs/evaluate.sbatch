#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Grade every prediction file, then copy the summary to project space (scratch is purged).
source "$REPO_DIR/cluster/killarney/common.sh"
activate extract
stamp=$(date +%Y%m%d-%H%M%S)
python -m llm_calc.extraction evaluate --pred-dir "$RUN_ROOT/preds" --out "$RUN_ROOT/eval/$stamp"
# the same figures and tables as docs/mock_results, from real predictions
python -m llm_calc.extraction figures --eval-dir "$RUN_ROOT/eval/$stamp" --out "$RUN_ROOT/eval/$stamp/report"
mkdir -p "$RESULTS_DIR"
cp -r "$RUN_ROOT/eval/$stamp" "$RESULTS_DIR/$stamp"
tar -C "$RUN_ROOT" -czf "$RESULTS_DIR/$stamp/preds.tar.gz" preds
echo "results: $RESULTS_DIR/$stamp"
