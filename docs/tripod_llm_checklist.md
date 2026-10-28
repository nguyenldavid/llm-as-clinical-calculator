# Reporting checklist: TRIPOD-LLM mapping

The extraction study will be reported under **TRIPOD-LLM** (Gallifant et al., *Nature Medicine*, 2025), the TRIPOD extension for studies that develop, tune or evaluate large language models in health care. This study evaluates existing models and fine-tunes encoders, so both the evaluation items and the development items apply.

The table maps each topic of the checklist to where this study answers it. Item numbers are omitted on purpose: copy them from the official checklist when the manuscript is assembled, so that a wording or numbering change in the guideline cannot silently misalign this file.

**Status key:**

- ✅ answered by the plan or the code now;
- 🟡 answered once the runs finish;
- ⬜ manuscript text still to write.

| Topic | Where it is answered | Status |
|---|---|---|
| **Title and abstract** | | |
| Identify the study as LLM evaluation, the task (information extraction for clinical scores) and the setting | Manuscript | ⬜ |
| Structured summary: objectives, data, models, outcomes, results, limitations | Manuscript | ⬜ |
| **Introduction** | | |
| Background and rationale; the gap (end-to-end score accuracy confounds extraction, knowledge and arithmetic) | Plan §1–2 | ✅ |
| Objectives and pre-specified hypotheses | Plan §3 (RQ1–RQ6, H1–H5) | ✅ |
| **Methods: data** | | |
| Data sources, provenance and licences (generator, Abacus-409, MedCalc-Bench Verified CC-BY-SA) | Plan §5; `medcalc.py` (pinned revision) | ✅ |
| Dates of data collection and of model access | Plan §11.2; `provenance.yaml`; API access dates recorded per run | 🟡 |
| Eligibility and exclusions, with counts | Flow figure (`figures.fig_flow`); `A409_match_report.yaml`, `T5_build_report.yaml` | ✅ |
| Data preparation: how vignettes were generated, perturbed and paraphrased; clinician review of T2/T3 | Plan §5.2; `generator.py`, `perturb.py`, `phrasings.py`, `paraphrase.py` | ✅ (T2/T3 review 🟡) |
| Train/test separation and leakage controls | Plan §5.3; `datasets.build_train` (test vignettes removed) | ✅ |
| Contamination of public test data in pre-training | Plan §9.5; `contamination.py` | ✅ method, 🟡 results |
| **Methods: outcome and reference standard** | | |
| Definition of the outcome (per-field values, final score, risk band) | Plan §8.1–8.2; `evaluate.py` | ✅ |
| How gold labels were produced and verified (generator metadata, OpenMedCalc round trip 10,000/10,000) | Plan §5.1; `gold.py`; `phase0_round_trip.yaml` | ✅ |
| Human reference: who, how blinded, agreement | Plan §9.4; `clinician.py`; κ per field | ✅ method, 🟡 results |
| Known gold-label issues and how they are handled | Plan §13 items 3 and 5; sensitivity S2, S4–S5 | ✅ |
| **Methods: models** | | |
| Each model: name, version or commit, provider, access mode, date | Table "Systems under comparison"; `models.tsv`, `models.lock.tsv`; Jev `jev-1.14.0` | ✅ (lock written at download) |
| Inference settings: decoding, temperature, max tokens, constrained decoding, hardware | `extractors/llm.py`, `models.tsv`; Plan §11.4 | ✅ |
| Prompts, exactly as used, including few-shot examples | `schemas.user_prompt`; few-shot sampling in `extractors/llm.py` (seeded); released with code | ✅ |
| Fine-tuning: data, objective, hyperparameters, selection | Plan §6B; `extractors/encoder.py`; TRAIN pool | ✅ |
| Output parsing and handling of invalid outputs | `evaluate.validate`; invalid counts as wrong (primary), S1 drops it | ✅ |
| **Methods: evaluation and statistics** | | |
| Metrics, with justification | Plan §8 | ✅ |
| Uncertainty: CIs and resampling unit (cluster bootstrap over base cases and notes) | Plan §9.1; `stats.cluster_bootstrap` | ✅ |
| Comparisons, tests and multiplicity (families i–iv, Holm) | Plan §9.1; Tables "Pairwise", "Clinician", "Variant" | ✅ |
| Modelling (GEE with cluster-robust SEs; system × set interaction) | Plan §9.1; `stats.gee_field_model` | ✅ |
| Sample size and precision | Plan §9.2; Table "Precision and minimum detectable differences" | ✅ |
| Calibration and selective prediction | Plan §8.5, §9.1; Tables "Calibration", "Operating points"; reliability diagram | ✅ |
| Sensitivity and subgroup analyses (pre-specified) | Plan §9.3; Tables "Sensitivity", "Subgroups" | ✅ |
| Robustness and safety (perturbations including prompt injection and omission) | Plan §5.2 (T4); Figure "Robustness" | ✅ |
| Run-to-run variability | Plan §9.1 | 🟡 |
| Error analysis and its reliability (two annotators, κ, adjudication) | Plan §10; `annotation.py` | ✅ method, 🟡 results |
| **Results** | | |
| Flow of data with numbers at each step | Flow figure | ✅ (T2/T3 🟡) |
| Characteristics of the test sets | Table "Test sets" | ✅ |
| Performance with CIs, overall and by set, calculator and field type | Figures "Headline", "Field heatmap", "Field type"; Tables "Primary", "Per calculator" | 🟡 |
| Comparisons with effect sizes | Forest plot; Table "Pairwise" | 🟡 |
| Errors: type, frequency and clinical impact (risk-band change) | Figure/Table "Error taxonomy"; Table "Clinical agreement" | 🟡 |
| Cost, latency and deployability | Figure "Cost"; Table "Operational metrics" | 🟡 |
| **Discussion** | | |
| Interpretation against the hypotheses; comparison with the original study | Figure "Paper arms" | ⬜ |
| Limitations: synthetic text, schema dependence, contamination, model churn, best-of-family selection | Plan §12 | ✅ draft |
| Implications for clinical use and the human-in-the-loop threshold | Plan §8.5 | ⬜ |
| **Other information** | | |
| Code, data and model availability | `REPRODUCE.md`; `MANIFEST.tsv`; T5 rebuild instructions (not redistributed, CC-BY-SA) | ✅ |
| Pre-registration (analysis plan frozen before grading real predictions) | Plan §13 item 9 | ⬜ |
| Ethics: no patient data in T1–T5; data governance for any real notes (T6) | Plan §5.2, §11.4 | ✅ |
| Funding, compute (Killarney / PAICE allocation) and competing interests | Manuscript | ⬜ |
| Use of LLMs in preparing the study (paraphrase generation; code assistance) | Plan §5.2; `paraphrase.py` | ⬜ statement |
