#!/bin/bash
# One-time setup on a Killarney LOGIN node: two virtualenvs in $ENV_DIR.
#   extract  CPU baselines, evaluation, encoder training (torch + transformers)
#   vllm     vLLM inference for LLMs, NuExtract and the paraphrase job
# Packages come from the Alliance wheelhouse (--no-index) whenever a wheel exists;
# anything missing falls back to PyPI and is reported.
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$ENV_DIR"
REQ_DIR="$(cd "$(dirname "$0")" && pwd)"

make_env() {  # make_env <name> <modules> <requirements file>
    local name=$1 modules=$2 req=$3
    echo "== $name: module load $modules"
    module load $modules
    if [ ! -d "$ENV_DIR/$name" ]; then
        virtualenv --no-download "$ENV_DIR/$name"
    fi
    source "$ENV_DIR/$name/bin/activate"
    pip install --no-index --upgrade pip
    while read -r pkg; do
        [ -z "$pkg" ] || [[ "$pkg" == \#* ]] && continue
        if ! pip install --no-index "$pkg"; then
            echo "!! no Alliance wheel for '$pkg'; installing from PyPI" >&2
            pip install "$pkg"
        fi
    done < "$req"
    pip freeze > "$ENV_DIR/$name.freeze.txt"
    deactivate
    module unload $modules || true
}

make_env extract "$MODULES_EXTRACT" "$REQ_DIR/requirements-extract.txt"
make_env vllm "$MODULES_VLLM" "$REQ_DIR/requirements-vllm.txt"

echo "Environments ready in $ENV_DIR (frozen requirements: $ENV_DIR/*.freeze.txt)"
