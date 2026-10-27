# Sourced at the top of every job script. Slurm copies batch scripts to a spool
# directory, so paths are taken from REPO_DIR (exported by submit.sh), not from
# the script's own location.
set -euo pipefail

if [ -z "${REPO_DIR:-}" ]; then
    echo "REPO_DIR is not set; submit jobs with cluster/killarney/submit.sh" >&2
    exit 2
fi
source "$REPO_DIR/cluster/killarney/config.sh"

activate() {  # activate <extract|vllm>
    # keep the default StdEnv; only add the Python (and, for vLLM, OpenCV) modules
    if [ "$1" = vllm ]; then module load $MODULES_VLLM; else module load $MODULES_EXTRACT; fi
    source "$ENV_DIR/$1/bin/activate"
}

# compute nodes: never reach out to the Hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT/preds" "$RUN_ROOT/models" "$RUN_ROOT/data" "$RUN_ROOT/logs"
cd "$REPO_DIR"

echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is); git $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# resolve a Hub id to its local snapshot (works offline once downloaded)
local_model() {
    python - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
PY
}

# test sets present in the repository (T2/T3 appear once reviewed sheets are built)
available_sets() {
    local sets=()
    for s in T1 A409 T2 T3 T4; do
        [ -f "$REPO_DIR/llm_calc/data/extraction/$s.jsonl" ] && sets+=("$s")
    done
    (IFS=,; echo "${sets[*]}")
}
SETS="${SETS:-$(available_sets)}"
TRAIN="$RUN_ROOT/data/train.jsonl"
train_arg() { [ -f "$TRAIN" ] && echo "--train $TRAIN" || true; }
