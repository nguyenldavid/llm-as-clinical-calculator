# Killarney (Alliance / PAICE) settings for the extraction benchmark.
# Edit the account, then `source cluster/killarney/config.sh` on a login node.
# Every value can also be overridden from the environment.

# AIP allocation (CCDB "aip-" RAP). Slurm reads SBATCH_ACCOUNT / SALLOC_ACCOUNT.
export KILLARNEY_ACCOUNT="${KILLARNEY_ACCOUNT:-aip-CHANGE_ME}"
export SBATCH_ACCOUNT="$KILLARNEY_ACCOUNT"
export SALLOC_ACCOUNT="$KILLARNEY_ACCOUNT"

# Project space is backed up and not purged; jobs cannot be submitted from $HOME.
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/$KILLARNEY_ACCOUNT/$USER}"
export REPO_DIR="${REPO_DIR:-$PROJECT_ROOT/llm-as-clinical-calculator}"
export ENV_DIR="${ENV_DIR:-$PROJECT_ROOT/envs}"

# Hugging Face cache. Compute nodes are treated as offline, so every model is
# downloaded on a login node first (download_models.sh). A 70B model is ~140 GB:
# check your project quota with `diskusage_report`, or point HF_HOME at $SCRATCH
# (purged after 60 days of inactivity).
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"

# Job outputs go to scratch (fast, large, purged after 60 days); finished
# evaluations are copied to RESULTS_DIR on project space by evaluate.sbatch.
export RUN_ROOT="${RUN_ROOT:-${SCRATCH:-$PROJECT_ROOT/scratch}/abacus-extract}"
export RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/abacus-extract-results}"

# Modules. The vLLM wheel in the Alliance wheelhouse needs python/3.13 and opencv
# (see the Alliance "VLLM" wiki page); confirm with `avail_wheels vllm` and
# `module spider python` before the first setup.
export MODULES_EXTRACT="${MODULES_EXTRACT:-python/3.11}"
export MODULES_VLLM="${MODULES_VLLM:-opencv/4.13 python/3.13}"

# Gated models (Llama, Gemma) need a token with the licence accepted on the Hub.
# Put `export HF_TOKEN=...` in ~/.hf_token (chmod 600); it is only used on login nodes.
if [ -f "$HOME/.hf_token" ]; then source "$HOME/.hf_token"; fi
