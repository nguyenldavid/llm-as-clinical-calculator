# Extraction benchmark: Phase 0 results and status

**Companion to:** [the analysis plan](extraction_analysis_plan.md)
**Code:** `llm_calc/extraction/`
**Cluster jobs:** [`cluster/killarney/`](../cluster/killarney/README.md)
**Tests:** `python -m pytest tests/test_extraction.py`

This report covers what has been built and run so far. It covers Phase 0 (per-field gold labels), the T1, A409, T4 and T5 test sets, the CPU baselines, and an analysis of the released Abacus-409 failures. T5 is the external set built from MedCalc-Bench Verified.

The GPU systems (encoders, NuExtract, LLMs) are implemented and were dry-run end to end. They have **not** been run for results yet; that is the Killarney step. T2 and T3 also wait on that step, followed by clinician review.

---

## 1. Status

| Plan item | State |
|---|---|
| Phase 0: gold mappers and round-trip check | **Done.** 10,000 / 10,000 cases agree. |
| Abacus-409 matched to per-field gold (A409) | **Done.** 397 / 397 non-empty rows. |
| T1 (1,000 cases) and T4 (1,869 perturbed cases) | **Done.** Committed under `llm_calc/data/extraction/`. |
| Training pool (disjoint from every test vignette) | **Done.** Generated on demand (`build-train`). |
| Variant R (raw-facts) schema | **Done for HAS-BLED.** Other calculators already take raw values (§2.4). |
| System A (rules), oracle, majority and template-inversion baselines | **Run on CPU** (§4). |
| System B (encoders) | Implemented. CPU smoke test only. |
| System C (NuExtract 2.0 / NuExtract3) | Implemented. CPU sanity check of NuExtract-2.0-2B only. |
| System D (vLLM, schema-constrained) | Implemented. Dry-run with a vLLM stand-in only. |
| System D (API models, e.g. GPT-4o) | Implemented (OpenAI-compatible client). Not run: no API keys in this environment. |
| T2 / T3 | Pipeline built. Waiting on the paraphrase job (Killarney), then clinician review. |
| T5 (MedCalc-Bench Verified, external notes) | **Done.** 637 notes for 7 shared calculators, built on demand (CC-BY-SA, not committed) (§4.1). |
| T6 (real notes) | Not started. Open decision in plan §13. |
| Figures and tables | **Mock shells for every planned experiment** in [mock_results/](mock_results/README.md). The same code renders the real page from Killarney output. |

## 2. Phase 0: gold labels

### 2.1 Round trip

Each calculator has a gold mapper that turns the sampled options into an OpenMedCalc request. Each mapper is an explicit table in `llm_calc/extraction/gold.py`, so a clinician can review it.

For 1,000 generated cases per calculator, `OpenMedCalc(gold request)` equals the ground truth the original experiment computed (`vignette.calculate_output_for_case`) in all **10,000** cases. The report is in `llm_calc/data/extraction/phase0_round_trip.yaml`.

### 2.2 Abacus-409 matched back to its seeds

The released vignettes come from the generator, so each one can be matched to the seed that produced it:

- All **397** non-empty rows match. The other 12 rows have an empty vignette.
- No match is ambiguous: when several seeds give the same text, they share the same gold request.
- The gold score equals the released `ground_truth` in all 397 cases.

Every released failure therefore now has per-field gold labels (`A409.jsonl`).

Matching needed one correction. `data.xlsx` was **edited after the experiment**: three CCI liver-disease options were reworded, which affected 47 released CCI vignettes. The old wordings are recorded in `generator.HISTORICAL_INPUT_STRINGS` and used only for matching.

### 2.3 Data-quality findings

1. **OpenMedCalc Wells DVT crashed on negative scores.** "Alternative diagnosis as likely" is worth −2, so the score can be −1 or −2. `calculate_wells_dvt` only set a risk category for 0, 1–2 and ≥3, so those scores raised `UnboundLocalError`. **Fixed** in `llm_calc/tools/openmedcalc/__init__.py`: a score ≤ 0 is low/unlikely. No released Abacus-409 failure has a negative Wells score, so the published error set is unaffected. The paper's OMC arm would have failed on any such case.
2. **A CCI scoring ambiguity in the released data.** The historical option "The patient reports cirrhosis due to chronic alcohol abuse and is awaiting liver transplant evaluation" was scored as moderate/severe liver disease (3 points). The Charlson definition requires portal hypertension or variceal bleeding for that category, so the text supports "mild" (1 point). The current wording adds "complicated by portal hypertension and variceal bleeding", which fixes it. This affects 13 A409 cases and needs clinician adjudication.
3. **Correlated random draws.** The generator seeds criterion *k* of case *s* with *s + k*, so consecutive case ids share random streams. Neighbouring cases in the original 10,000 trials are therefore not independent draws. New sets space seeds 1,000 apart.
4. **Typos in the model inputs:**
   - GAD-7 template 2 contains `{gad7_feeling_afraid}}`, "CThe overall impact" and "an you calculate".
   - The ARISCAT option misspells the antibiotic as "azirthromycin".
   - Templates ask for the "Charleston" Comorbidity Index.
   - One CCI option has a trailing space.
   - Option id 229 appears twice in `data.xlsx`.

   They are kept, because they are part of what the original models saw.
5. **The OpenMedCalc request models enforce physiological ranges** (for example pH < 8.4, temperature < 50 °C). An extractor that places a number in the wrong field gets a schema-validation failure instead of a silently wrong score. This is the type-safety property the plan relies on, and the evaluator counts such outputs as invalid.
6. **OpenMedCalc SOFA under-scored ventilated patients with a P/F ratio of 200–399.** These patients got 0 respiratory points instead of 2 (200–299) or 1 (300–399): `calculate_sofa` only scored ventilated patients below 200. This was found by cross-checking against MedCalc-Bench, where all 22 SOFA disagreements were exactly this case. **Fixed.** The paper's ventilated vignettes all have P/F < 200, so its ground truth is unaffected.
7. **OpenMedCalc counts HAS-BLED "elderly" as age ≥ 65.** The original HAS-BLED definition and MDCalc use > 65. Two MedCalc-Bench patients aged exactly 65 disagree by one point because of this. No vignette in the paper is 65. **Not changed**; it needs a clinician's decision.
8. **The type-safe schema can reject real patients.** The MELD-Na request model requires 110 < Na < 160, so 15 MedCalc-Bench patients with severe hypo- or hypernatraemia cannot be expressed, although MELD-Na clamps sodium to 125–137 anyway. Two MedCalc-Bench rows have GCS 1 or 2, which is impossible (minimum 3); these are annotation errors. All 17 rows are excluded from T5 and listed in `T5_build_report.yaml`.
9. **The OpenMedCalc web app cannot start from a fresh clone.** `llm_calc/tools/openmedcalc/calculator_docs/` is git-ignored, and every calculator reads its markdown. The benchmark therefore loads the calculator functions directly from source, without the web app (`extraction/omc.py`).

### 2.4 Schema variants

The decision schema (D) and the raw-facts schema (R) differ only where a threshold is folded into a boolean. Of the ten OpenMedCalc request models, most already take raw values:

- MELD-Na, SOFA and PSI/PORT take labs and vitals directly.
- ARISCAT and CCI take age and SpO₂ directly.

**HAS-BLED is the clear exception.** Its fields `hypertension` (SBP > 160), `severe_renal_disease` (Cr > 2.26, dialysis or transplant) and `alcohol` (≥ 8 drinks/week) are all thresholds. Variant R is implemented for HAS-BLED: `RawRequestHASBLED` plus `to_decision()`. On 1,000 cases, its gold converts exactly to the decision gold.

## 3. Test sets

| Set | n | Built from |
|---|---|---|
| T1 | 1,000 (100 per calculator) | Original generator, fresh seeds |
| A409 | 397 | Released failures, matched to seeds |
| T4 | 1,869 | T1 cases × 6 perturbations: distractors 452, long note 485, abbreviations 332, reorder 250, units 200, conflicting values 150 |
| T5 | 637 | MedCalc-Bench Verified (pinned revision `58fc3de`), test and train splits, 7 shared calculators |
| TRAIN | about 19,000 (2,000 per calculator, minus duplicates of test vignettes) | Generator, separate seed range |

Some perturbations apply only to some calculators. Units, for example, apply only where labs are given. Every T4 case keeps its T1 gold labels, and `meta.base_id` links it back for paired comparison.

## 4. CPU baselines

Score accuracy is computed by OpenMedCalc from the extracted request. A field is correct when it is exact, or within 1% for floats. Full tables are in `llm_calc/data/extraction/results_cpu_baselines/`.

| System | T1 score acc | A409 score acc | T4 score acc | T4 macro field acc | **T5 score acc** | **T5 macro field acc** |
|---|---|---|---|---|---|---|
| Oracle (harness check) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Majority class per field (floor) | 0.067 | 0.033 | 0.051 | 0.467 | 0.053 | 0.450 |
| Template inversion (memorisation ceiling) | **1.000** | **1.000** | **0.063** | 0.440 | 0.077 | 0.377 |
| Rules, System A (*in-sample on T1/T4*) | 1.000 | 0.967 | 1.000 | 1.000 | **0.155** | **0.737** |

**Template inversion settles the T1 question.** It knows the templates and option phrasings and nothing clinical, yet it solves all of T1 and every one of the 397 vignettes that GPT-4o and Llama 3.1 got wrong. As soon as the wording changes (T4), it falls to the majority floor. So T1 and A409 measure whether a system can read these particular sentences, not whether it can extract clinical criteria. This confirms the plan's decision to make T2 and T3 the primary sets.

**The rules results are in-sample by construction.** I wrote the rules after reading the T1 phrasings and the T4 perturbation types (which I also wrote), and fixed every T1/T4 error I found. Those fixes include general ones: negation scope across lists, skipping the question sentence, family-history attribution, and number words.

- Their A409 result (0.967) reflects a single disagreement, the CCI ambiguity in §2.3(2). The rules apply the Charlson definition.
- Their T1/T4 results show how far a developer who can see the data can push hand-written rules. They do not estimate how rules perform on unseen text.
- T2 and T3, written by a different model family and reviewed by a clinician, are where System A gets a fair test. So is T5 (§4.1).

### 4.1 T5: the first out-of-sample result

T5 uses MedCalc-Bench Verified notes: case-report excerpts and synthetic notes written by other people, with extracted parameters annotated. The seven calculators shared with this benchmark give 654 rows. 637 map to valid OpenMedCalc requests (§2.3.8).

- **The mapping is checked against MedCalc-Bench's own answers.** OpenMedCalc's score on the mapped request agrees with MedCalc-Bench's answer in **629 of 637** cases, after the SOFA fix. Of the remaining eight, six are MELD-Na rounding or dialysis-handling differences and two are the HAS-BLED age-65 rule.
- **Undocumented criteria are recorded.** Criteria that MedCalc-Bench did not annotate (21% of field instances) are marked "not documented" in `field_sources`. The gold value for these is the calculator default.
- **The rules system was not tuned on T5.** I had not read the T5 notes while writing it and will not tune it on them, so this is a genuine out-of-sample estimate:
  - **Score accuracy:** 15.5% [12.9%, 18.6%], against 100% on T1/T4 and a 5.3% majority floor.
  - **Macro field accuracy:** 0.737, against 0.450 for the majority floor.
  - **Undocumented fields:** 96% correct (mostly "absent").
  - **Documented fields:** 73% correct.
  - **Validity:** 2.7% of its requests are rejected by the schema's range checks.
- **Where the rules fail:** mostly on numbers phrased differently from the templates ("partial pressure of oxygen", "BP 128/76", "FiO2 of 40%"), and on SOFA's MAP. MedCalc-Bench gives systolic and diastolic pressure, so the MAP has to be *computed* from them. A pure extractor cannot fill a derived field from text alone, which is another case for raw-facts schemas.
- **Template inversion falls to the floor (7.7%)** on external notes, as expected.

Per-calculator rules results on T5:

| Calculator | Score acc | Macro field acc |
|---|---|---|
| HAS-BLED | 0.40 | 0.69 |
| CCI | 0.25 | 0.89 |
| Wells DVT | 0.19 | 0.74 |
| PSI | 0.12 | 0.74 |
| MELD-Na | 0.06 | 0.69 |
| Caprini | 0.05 | 0.87 |
| SOFA | 0.03 | 0.53 |

Many-field scores (Caprini has 32 fields) can be wrong on the final score even when most fields are right.

## 5. What final-score errors reveal: Abacus-409 attribution

For each released failure with a numeric answer (373 of 397), the analysis searched for the smallest number of changes to the gold request that make OpenMedCalc return the model's wrong answer. Alternative values come from the generated option space. Output: `llm_calc/data/extraction/a409_attribution.md`.

| Fields changed to reproduce the wrong answer | n | % of 397 |
|---|---|---|
| 1 | 293 | 73.8 |
| 2 | 45 | 11.3 |
| more than 2 | 35 | 8.8 |
| no numeric answer | 24 | 6.0 |

By annotated error class, a single field change reproduces:

| Error class | Share reproduced by one field change |
|---|---|
| Interpretation Error | 89% |
| Incorrect Criteria | 80% |
| Assignment Error | 71% |
| Calculation Error | 67% |
| Incorrect Formula | 54% (another 25% have no numeric answer) |

**The control matters.** If each error is mirrored (same size, opposite direction), 66% of them can still be reproduced by one field change. For the additive scores, the mirrored rate is about the same as the observed one, so a single-field explanation is no evidence there:

| Calculator | Observed | Mirrored |
|---|---|---|
| NIHSS | 100% | 98% |
| SOFA | 93% | 93% |
| HAS-BLED | 80% | 80% |
| Caprini | 80% | 78% |

Only ARISCAT (58% observed vs 0% mirrored) and MELD-Na (50% vs 25%) separate clearly.

**Conclusion.** The final score cannot tell you which field an LLM misread. The released error classes cannot be recovered from answers either, except for a few non-additive calculators. That is the empirical case for grading extraction per field against typed gold labels, rather than inferring it from final-score errors.

## 6. Smoke tests (pipeline checks, not results)

- **Encoder** (`prajjwal1/bert-tiny`, CPU, 4 epochs on 566 cases):
  - Trains, predicts and evaluates for both variants.
  - Numeric span heads reach about 98% after only a few epochs.
  - Some predictions are rejected by the schema's range checks (for example sodium placed in pH), as intended.
  - The real encoders run on Killarney.
- **NuExtract-2.0-2B** (transformers, CPU, 6 T1 cases):
  - The template renders as NuMind documents it.
  - 5 of 6 cases had every field correct.
  - The sixth produced malformed JSON (`"true}`). This led to the optional `--constrained` mode (JSON-schema decoding over NuExtract's own template). `models.tsv` runs NuExtract3 both natively and constrained.
- **vLLM and NuExtract jobs:** dry-run end to end through the Slurm job scripts with a stand-in for vLLM that has the same call signatures. The call signatures were checked against vLLM's current source (`LLM.chat(..., chat_template_kwargs=...)`, `StructuredOutputsParams(json=...)`).

## 7. Next steps

1. **Killarney.** Follow [`cluster/killarney/README.md`](../cluster/killarney/README.md):
   - set the `aip-` account;
   - run `setup_envs.sh` and `download_models.sh` on a login node;
   - then `submit.sh all`.

   This runs Phase 0 again, the CPU baselines, 4 encoders, 6 NuExtract configurations and 7 LLM configurations (including the paper's Llama 3.1 70B), then evaluates.
2. **T2/T3.** Run `submit.sh paraphrase options` and `submit.sh paraphrase vignettes` (Mistral-Small-24B, a family not under test), have a clinician review the sheets, then run `build-t2` and `build-t3`.
3. **API arms.** Run GPT-4o through `predict --system api` on the synthetic sets only. Re-run the paper's original end-to-end arms on T1–T4 with the existing `llmcalc experiment` code for the RQ2 comparison.
4. **Clinician decisions:**
   - the CCI liver-disease adjudication (§2.3.2);
   - risk-band cut-offs (`evaluate.RISK_BANDS` are provisional);
   - the T5/T6 scope (plan §13).
