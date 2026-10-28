#!/bin/bash
# Download every model in models.tsv into $HF_HOME. Run on a LOGIN node:
# compute nodes are used offline (HF_HUB_OFFLINE=1), as the Alliance docs require.
#   ./download_models.sh            all models
#   ./download_models.sh nuextract  only rows whose kind matches
#
# models.lock.tsv pins each model to a Hub commit. The first download records the
# commit it got; later downloads (another user, a re-run) fetch exactly that commit.
# Commit models.lock.tsv with the results it produced.
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
    python - "$model" "$HERE/models.lock.tsv" <<'PY'
import csv
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model, lock_path = sys.argv[1], Path(sys.argv[2])
lock = {}
if lock_path.exists():
    lock = {r["model"]: r["revision"] for r in csv.DictReader(open(lock_path), delimiter="\t")}
path = snapshot_download(model, revision=lock.get(model),
                         ignore_patterns=["original/*", "*.pth", "consolidated*", "*.gguf"])
revision = Path(path).name  # snapshot directories are named by commit hash
print("   ->", path, "(pinned)" if model in lock else "(recorded in models.lock.tsv)")
if model not in lock:
    lock[model] = revision
    with open(lock_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["model", "revision"])
        w.writerows(sorted(lock.items()))
PY
done
du -sh "$HF_HOME" || true
