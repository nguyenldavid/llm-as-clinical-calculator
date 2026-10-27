#!/bin/bash
# Download every model in models.tsv into $HF_HOME. Run on a LOGIN node:
# compute nodes are used offline (HF_HUB_OFFLINE=1), as the Alliance docs require.
#   ./download_models.sh            all models
#   ./download_models.sh nuextract  only rows whose kind matches
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
module load $MODULES_EXTRACT
source "$ENV_DIR/extract/bin/activate"
mkdir -p "$HF_HOME"

filter="${1:-}"
grep -v '^#' "$HERE/models.tsv" | awk -F'\t' 'NF>=2 {print $1"\t"$2}' | sort -u | while IFS=$'\t' read -r kind model; do
    if [ -n "$filter" ] && [ "$kind" != "$filter" ]; then continue; fi
    echo "== $kind $model"
    # weights only: skip duplicate original/ or consolidated checkpoints where a repo ships both
    python - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], ignore_patterns=["original/*", "*.pth", "consolidated*", "*.gguf"])
print("   ->", path)
PY
done
du -sh "$HF_HOME" || true
