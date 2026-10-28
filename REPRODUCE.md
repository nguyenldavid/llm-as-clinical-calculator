# Reproducing the extraction benchmark

This is the path from a fresh clone to every number, figure and table in the
extraction study. The design and statistics are in
[docs/extraction_analysis_plan.md](docs/extraction_analysis_plan.md); the
original agent study (Goodell et al., 2025) is reproduced with the instructions
in [README.md](README.md).

Steps marked **CPU** run on a laptop. Steps marked **GPU** run on Killarney
(see [cluster/killarney/README.md](cluster/killarney/README.md)); any Slurm
cluster with L40S/H100-class GPUs works after editing `config.sh`.

## 0. Environment

```bash
git clone https://github.com/nguyenldavid/llm-as-clinical-calculator.git
cd llm-as-clinical-calculator
python3 -m venv .venv && source .venv/bin/activate
pip install -r cluster/killarney/requirements-extract.txt   # CPU parts: pydantic, numpy, pandas, statsmodels, matplotlib
python -m pytest tests/test_extraction.py -q                 # every check below in under a minute
```

The benchmark CLI is `python -m llm_calc.extraction <command>`. It never imports
the web app or needs API keys, so it runs offline.

## 1. Datasets (CPU)

The committed sets are generated deterministically; rebuilding them must give
identical bytes.

```bash
python -m llm_calc.extraction build-gold --n 1000     # Phase 0: gold mappers vs the original ground truth (10,000/10,000)
python -m llm_calc.extraction build-sets              # T1, A409, T4
python -m llm_calc.extraction build-t5                # T5 from MedCalc-Bench Verified, pinned revision (needs internet)
python -m llm_calc.extraction manifest --check        # SHA-256 of every set against llm_calc/data/extraction/MANIFEST.tsv
```

- **T5** is derived from MedCalc-Bench Verified (CC-BY-SA-4.0) and is not
  redistributed here. `build-t5` downloads the pinned revision and `manifest --check`
  confirms the rebuild matches ours.
- **T2 and T3** need the paraphrase job (step 3) and clinician approval of each
  candidate. They are then built with `build-t2` / `build-t3` from the approved
  review sheets, which are committed alongside the sets.

## 2. CPU baselines

```bash
for s in oracle majority template_inversion rules; do
  python -m llm_calc.extraction predict --system $s --sets T1,A409,T4,T5 --variants D,R --out preds_cpu/
done
python -m llm_calc.extraction evaluate --pred-dir preds_cpu/ --out eval_cpu/
```

The committed summaries are in `llm_calc/data/extraction/results_cpu_baselines/`.
The oracle must score 100%, which checks the evaluator itself.

## 3. GPU systems (Killarney)

```bash
cluster/killarney/setup_envs.sh          # once, login node
cluster/killarney/download_models.sh     # login node; pins each model in models.lock.tsv
cluster/killarney/submit.sh paraphrase options      # T2 candidates -> clinician review
cluster/killarney/submit.sh paraphrase vignettes    # T3 candidates -> clinician review
cluster/killarney/submit.sh all          # prepare, encoders, NuExtract, LLMs, then evaluate
cluster/killarney/submit.sh contamination           # completion probe, T5 vs T3 (plan §9.5)
```

Each model is pinned to the Hub commit recorded in
`cluster/killarney/models.lock.tsv`. Commit that file with the results.

## 4. Hosted systems

Only synthetic sets and T5 may be sent to external APIs.

```bash
export TYPESAFE_API_KEY=...   # Jev, pinned to jev-1.14.0
python -m llm_calc.extraction predict --system jev --model jev-1.14.0 --sets T1,A409,T2,T3,T4,T5 --variants D,R --out preds/
export OPENAI_API_KEY=...     # optional API LLM arm, pinned to a dated snapshot
python -m llm_calc.extraction predict --system api --model gpt-4o-2024-08-06 --sets T1,A409,T2,T3,T4,T5 --variants D,R --out preds/
```

## 5. Human arms

```bash
# clinician reference (plan §9.4): sheets out, two readers fill them blind, predictions in
python -m llm_calc.extraction clinician sample --out sheets/clinician
python -m llm_calc.extraction clinician import --sheets sheets/clinician/reader_A --reader A --out preds/
python -m llm_calc.extraction clinician import --sheets sheets/clinician/reader_B --reader B --out preds/
python -m llm_calc.extraction clinician agree --a preds/clinician_A/T2_D.jsonl --b preds/clinician_B/T2_D.jsonl --out eval/

# field-error annotation (plan §10)
python -m llm_calc.extraction annotate sample --eval-dir eval/ --out sheets/errors
python -m llm_calc.extraction annotate agree --a sheets/errors/annotator_A.tsv --b sheets/errors/annotator_B.tsv --out sheets/errors
python -m llm_calc.extraction annotate count --sheet sheets/errors/agreed.tsv --sheet sheets/errors/adjudicate.tsv --key sheets/errors/key.tsv --out eval/
```

## 6. Statistics, figures and tables

```bash
python -m llm_calc.extraction evaluate --pred-dir preds/ --out eval/
python -m llm_calc.extraction figures --eval-dir eval/ --out report/
```

- **What you get:** `report/README.md` holds every figure and table, numbered in page order.
- **Figures:** `report/figures/` has vector PDFs and 300-dpi PNGs at 89/183 mm widths.
- **Source data:** `report/source_data/` has the data behind every figure and table.
- **Provenance:** on Killarney, the `evaluate` job also writes `provenance.yaml` (git commit, date), the dataset manifest, the model lockfile and `pip freeze`.

Fixed choices:

- **Bootstrap:** every interval uses 2,000 resamples with a fixed seed (`stats.N_BOOT`, `seed=0`), so the numbers are reproducible exactly.
- **Decoding:** LLMs decode greedily.
- **Repeats:** where the plan calls for 3 repeats, the seed is part of the system name.

### Mock page

[docs/mock_results/](docs/mock_results/README.md) shows every planned figure and
table on mock predictions. It is the analysis plan in executable form:

```bash
python -m llm_calc.extraction mock-results --out /tmp/mock_eval
python -m llm_calc.extraction figures --eval-dir /tmp/mock_eval --out docs/mock_results --mock
```
