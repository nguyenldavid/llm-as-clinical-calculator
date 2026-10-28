# Running the extraction benchmark on Killarney

[Killarney](https://docs.alliancecan.ca/wiki/Killarney) is the PAICE AI cluster run by the Vector Institute and SciNet. It has two node types:

- **Standard nodes:** 168 nodes, each with 4 × L40S 48 GB GPUs, 64 cores and 512 GB RAM.
- **Performance nodes:** 10 nodes, each with 8 × H100 80 GB GPUs.

Jobs use Slurm under an `aip-` allocation. This folder contains everything needed to run the GPU parts of [the analysis plan](../../docs/extraction_analysis_plan.md):

- encoder fine-tuning;
- NuExtract;
- LLMs with schema-constrained decoding (vLLM);
- the paraphrase job that produces T2/T3 candidates.

The CPU parts run on the cluster too: the Phase 0 check, the training pool and the baselines.

| File | Purpose |
|---|---|
| `config.sh` | Account, paths, modules. **Edit `KILLARNEY_ACCOUNT` first.** |
| `setup_envs.sh` | One-time setup on a login node. Creates the `extract` and `vllm` virtualenvs from Alliance wheels. |
| `download_models.sh` | Login node only. Downloads every model in `models.tsv` into `$HF_HOME`. |
| `models.tsv` | What runs: system kind, Hub id, GPU type and count, time limit, extra `predict` arguments. |
| `submit.sh` | Submits jobs and chains them with dependencies. |
| `jobs/*.sbatch` | Job bodies (`prepare`, `encoder`, `nuextract`, `vllm`, `paraphrase`, `contamination`, `evaluate`). |
| `models.lock.tsv` | Written by `download_models.sh`: the Hub commit each model is pinned to. Commit it with the results. |

## Cluster rules these scripts follow

From the Alliance documentation:

- **Accounts:** every job needs an `aip-` account. `config.sh` exports it as `SBATCH_ACCOUNT`.
- **Time limits:** every job states `--time`. GPUs are requested as `--gpus-per-node=l40s:N` or `h100:N`.
- **No submission from `$HOME`:** `submit.sh` refuses to run from there. Clone the repository into project space.
- **Downloads on login nodes:** models are downloaded on a login node. Jobs set `HF_HUB_OFFLINE=1`, so a missing model fails fast rather than hanging.
- **Python packages:** they come from the Alliance wheelhouse (`virtualenv --no-download`, `pip install --no-index`). There is no Conda.
- **vLLM:** the vLLM wheel is built against `python/3.13` with `opencv`, per the Alliance vLLM page. Check `avail_wheels vllm` and pin the version in `requirements-vllm.txt` if you need reproducibility.
- **Storage:**
  - Outputs are written to `$SCRATCH`, which is purged after 60 days.
  - `evaluate.sbatch` copies each finished evaluation, plus a tarball of the predictions, to `RESULTS_DIR` on project space.
  - Models default to project space (`HF_HOME`). The 70B models are about 140 GB each, so check `diskusage_report` first.

## One-time setup (login node)

```bash
ssh killarney.alliancecan.ca
cd ~/projects/aip-<pi>/$USER            # project space
git clone https://github.com/nguyenldavid/llm-as-clinical-calculator.git
cd llm-as-clinical-calculator && git checkout claude/nice-brahmagupta-5o8eai
$EDITOR cluster/killarney/config.sh     # set KILLARNEY_ACCOUNT=aip-<pi>

# gated models (Llama 3.1): accept the licence on huggingface.co, then
echo 'export HF_TOKEN=hf_...' > ~/.hf_token && chmod 600 ~/.hf_token

module spider python; avail_wheels vllm # confirm the module versions in config.sh
cluster/killarney/setup_envs.sh
cluster/killarney/download_models.sh    # or: download_models.sh nuextract

# T5 (MedCalc-Bench Verified) needs internet, so build it here, not in a job
module load $MODULES_EXTRACT && source "$ENV_DIR/extract/bin/activate"
PYTHONPATH=$PWD python -m llm_calc.extraction build-t5   # writes llm_calc/data/extraction/T5.jsonl (git-ignored)
deactivate
```

## Running

```bash
cd ~/projects/aip-<pi>/$USER/llm-as-clinical-calculator
DRY_RUN=1 cluster/killarney/submit.sh all   # print the sbatch commands only
cluster/killarney/submit.sh all             # prepare -> every GPU row -> evaluate
sq                                          # your jobs
```

You can also submit stages separately. Rows can be filtered by a substring of the model id:

```bash
cluster/killarney/submit.sh prepare
PREP_ID=<prepare job id> cluster/killarney/submit.sh encoder
cluster/killarney/submit.sh nuextract NuExtract3
cluster/killarney/submit.sh vllm Llama-3.1-70B
cluster/killarney/submit.sh evaluate
```

Before a long run, do a quick pilot. Add `--limit 20` to a row's extra arguments in `models.tsv`, or run on a single GPU interactively:

```bash
salloc --time=1:00:00 --gpus-per-node=l40s:1 --cpus-per-task=12 --mem=100G
# on the compute node (common.sh uses `set -e`, so it is not sourced in an interactive shell)
source cluster/killarney/config.sh
module load $MODULES_VLLM && source "$ENV_DIR/vllm/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
python -m llm_calc.extraction predict --system nuextract --model numind/NuExtract3 --sets T1 --limit 20 \
    --out "$RUN_ROOT/preds-pilot"
```

## What each job does

| Job | Resources | Steps |
|---|---|---|
| `prepare` | CPU, 8 cores, 2 h | Phase 0 round-trip check (fails the job if any case disagrees); training pool of 2,000 cases per calculator, never overlapping a test vignette; oracle, majority, template-inversion and rules baselines on every available set. |
| `encoder` | 1 × L40S | Fine-tunes one shared encoder with a head per schema field (`train-encoder`), then predicts. |
| `nuextract` | 1 × L40S | NuExtract with its native JSON template. `--few-shot 3` adds in-context examples; `--constrained` adds JSON-schema decoding. |
| `vllm` | 1–4 × L40S | Chat LLM with JSON-schema constrained decoding. 70B models use tensor parallelism across the 4 GPUs of one standard node. |
| `paraphrase` | 2 × L40S | Writes T2 (`options`) or T3 (`vignettes`) candidate sheets for clinician review. |
| `evaluate` | CPU | Grades everything under `$RUN_ROOT/preds`, renders the figures and tables (the real version of `docs/mock_results/`), and copies the results to project space. |

Resources per row are set in `models.tsv`. Each GPU gets 12 cores and 100 GB of host memory, which fits within a standard node's 16 cores and 128 GB per GPU.

## T2/T3: the human-review step

```bash
cluster/killarney/submit.sh paraphrase options     # -> $RUN_ROOT/data/review_options_<model>.tsv
cluster/killarney/submit.sh paraphrase vignettes   # -> $RUN_ROOT/data/review_vignettes_<model>.tsv
```

1. A clinician fills the `approved` column (`y`/`n`) and, optionally, `note`.
2. On a login node, build the sets:

```bash
python -m llm_calc.extraction build-t2 --review-sheet reviewed_options.tsv
python -m llm_calc.extraction build-t3 --sheet reviewed_vignettes.tsv
```

`build-t2` and `build-t3` write `T2.jsonl` and `T3.jsonl` next to T1. From then on, every job also evaluates on them. To train encoders on the *train*-split phrasings, set `REVIEW_SHEET=<approved sheet>` when you submit `prepare`. Test-split phrasings never enter training.

## Models outside Killarney

The paper's GPT-4o arm, and any other API model, run through the OpenAI-compatible client:

```bash
python -m llm_calc.extraction predict --system api --model gpt-4o --base-url https://api.openai.com/v1 \
    --api-key-env OPENAI_API_KEY --sets T1,A409,T4 --out preds
```

Only send synthetic sets (T1–T4, A409) to external APIs. Real notes (T6) stay on Killarney unless the data use agreement explicitly allows otherwise.
