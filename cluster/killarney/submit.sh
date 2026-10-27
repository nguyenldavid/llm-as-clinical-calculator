#!/bin/bash
# Submit benchmark jobs on Killarney. Run from a login node, inside project space.
#
#   ./submit.sh prepare                 CPU: Phase 0 check, training pool, CPU baselines
#   ./submit.sh encoder [model-substr]  fine-tune + predict each encoder in models.tsv
#   ./submit.sh nuextract [substr]      NuExtract rows
#   ./submit.sh vllm [substr]           LLM rows
#   ./submit.sh paraphrase [options|vignettes]
#   ./submit.sh evaluate                grade everything under $RUN_ROOT/preds
#   ./submit.sh all                     prepare, then every GPU row, then evaluate
#
# GPU jobs that need the training pool wait for the prepare job (afterok).
# Add DRY_RUN=1 to print the sbatch commands without submitting.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
export REPO_DIR

if [[ "$KILLARNEY_ACCOUNT" == *CHANGE_ME* ]]; then
    echo "Set KILLARNEY_ACCOUNT (your aip- allocation) in cluster/killarney/config.sh" >&2
    exit 2
fi
case "$PWD/" in "$HOME"/projects/*|"$PROJECT_ROOT"/*|/project/*|/scratch/*) ;;
    "$HOME"/*) echo "Killarney does not accept jobs submitted from \$HOME; cd into project space" >&2; exit 2 ;;
esac
mkdir -p "$RUN_ROOT/logs"

sb() {  # sb <sbatch args...>: prints the job id
    local cmd=(sbatch --parsable --chdir="$REPO_DIR" --output="$RUN_ROOT/logs/%x-%j.out" "$@")
    if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}" >&2; echo "DRY$RANDOM"; else "${cmd[@]}"; fi
}

PREP_ID="${PREP_ID:-}"
dep() { [ -n "$PREP_ID" ] && echo "--dependency=afterok:$PREP_ID" || true; }

submit_prepare() {
    PREP_ID=$(sb --job-name=abx-prepare "$HERE/jobs/prepare.sbatch")
    echo "prepare: $PREP_ID"
}

submit_rows() {  # submit_rows <kind> [substring]
    local kind=$1 filter=${2:-} ids=()
    while IFS=$'\t' read -r k model gpu gpus time extra; do
        [[ -z "$k" || "$k" == \#* || "$k" != "$kind" ]] && continue
        [[ -n "$filter" && "$model" != *"$filter"* ]] && continue
        local name="abx-$kind-$(basename "$model")"
        local mem=$(( gpus * 100 ))G cpus=$(( gpus * 12 ))
        local id
        # exported inside the command substitution's subshell, so nothing leaks between rows
        id=$(export MODEL="$model" EXTRA="$extra" KIND="$kind"
             sb --job-name="$name" --gpus-per-node="$gpu:$gpus" --cpus-per-task="$cpus" --mem="$mem" \
                --time="$time" $(dep) --export=ALL "$HERE/jobs/$kind.sbatch")
        echo "$kind $model ($gpu x$gpus, $time) $extra: $id"
        ids+=("$id")
    done < "$HERE/models.tsv"
    LAST_IDS+=("${ids[@]}")
}

submit_paraphrase() {
    local mode=${1:-options}
    while IFS=$'\t' read -r k model gpu gpus time extra; do
        [[ "$k" != paraphrase ]] && continue
        echo "paraphrase $mode $model: $(export MODEL="$model" EXTRA="$extra" MODE="$mode"
            sb --job-name="abx-paraphrase-$mode" --gpus-per-node="$gpu:$gpus" --cpus-per-task=$(( gpus * 12 )) \
               --mem=$(( gpus * 100 ))G --time="$time" --export=ALL "$HERE/jobs/paraphrase.sbatch")"
    done < "$HERE/models.tsv"
}

submit_evaluate() {
    local after=""
    if [ ${#LAST_IDS[@]} -gt 0 ]; then after="--dependency=afterany:$(IFS=:; echo "${LAST_IDS[*]}")"; fi
    echo "evaluate: $(sb --job-name=abx-evaluate $after "$HERE/jobs/evaluate.sbatch")"
}

LAST_IDS=()
case "${1:-}" in
    prepare) submit_prepare ;;
    encoder|nuextract|vllm) submit_rows "$1" "${2:-}" ;;
    paraphrase) submit_paraphrase "${2:-options}" ;;
    evaluate) submit_evaluate ;;
    all) submit_prepare; submit_rows encoder; submit_rows nuextract; submit_rows vllm; submit_evaluate ;;
    *) sed -n '2,14p' "$0"; exit 1 ;;
esac
