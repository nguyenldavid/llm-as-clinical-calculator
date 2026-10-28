# Analysis Plan: Benchmarking Clinical Score Extraction Against a Typed Calculator API

**Status:** Phase 0 complete. T1, A409, T4 and T5 (MedCalc-Bench Verified) are built, and the CPU baselines have been run. The GPU stages are ready to run on Killarney (§11.4). Results so far are in [extraction_phase0_results.md](extraction_phase0_results.md). Mock versions of every planned figure and table are in [mock_results/](mock_results/README.md).
**Builds on:** Goodell et al., *Large Language Model Agents Can Use Tools to Perform Clinical Calculation*, NPJ Digital Medicine (2025), and the code and data in this repository.

---

## 1. Summary

The original study asks an end-to-end question: given a clinical vignette, does a model return the correct score? It finds that task-specific tools (OpenMedCalc) help the most. A score error can come from three different steps, though:

1. **Extraction.** Reading the text into the calculator's inputs: is the patient on dialysis, what is the creatinine, is the left arm drifting.
2. **Knowledge.** Knowing which criteria and formula the calculator uses.
3. **Arithmetic.** Computing the result.

A typed calculator API removes (2) and (3) by design. This plan separates **extraction** from **computation** and asks how well different families of models fill a typed, validated request schema from clinical text:

- rule-based clinical NLP
- BERT-style encoders
- small dedicated extraction models such as NuExtract
- large language models (LLMs) with schema-constrained decoding

A deterministic API computes the score in every case, so all systems are compared on the same footing. They are graded **per field**, not only on the final number.

---

## 2. Background: what the existing study tests

### 2.1 Inputs

The inputs are synthetic natural-language vignettes that end with a question such as *"Can you calculate their HAS-BLED score?"*.

- **Templates.** Each calculator has a few templates in `llm_calc/data/build_database/vignette_templates/*.yaml`, with placeholders such as `{hasbled_hypertension}`.
- **Criterion options.** Each placeholder is filled by sampling one row from the `criterion_option` sheet of `llm_calc/data/build_database/data.xlsx`. Each row holds a fixed phrase (`input_string`) and its point value (`score_effect`). For example:
  - `"They report drinking nine alcoholic drinks per week"` gives `hasbled_alcohol` +1.
  - `"Patient expresses abstinence from alcohol"` gives +0.
- **Generation.** `llm_calc/lib/vignette.py` (`build_single_vignette`) picks a template, samples one option per criterion with a fixed seed, and substitutes the phrases.
- **Ground truth.** `calculate_output_for_case` computes the answer deterministically from the sampled options, either by summing `score_effect` or by calling the OpenMedCalc function for formula-based scores such as MELD-Na.

### 2.2 Arms and models

| Arm | Description |
|---|---|
| `*_base` | LLM alone (system prompt, one worked example, vignette) |
| `*_rag` | Retrieval over MDCalc-derived reference text (`external_data/md`) |
| `*_ci` | Code interpreter tool |
| `*_rag_ci` | RAG plus code interpreter |
| `*_omc` | Agentic tool calls to the typed OpenMedCalc calculators |

The models were GPT-4o and Llama 3.1 (via OpenRouter). A second "parser" LLM turns each free-text reply into `{answer, explanation, able_to_answer}` (`llm_calc/tools/structured_output.py`), and `answer` is compared with the ground truth.

### 2.3 Released datasets

| Dataset | Contents | Limitation for this plan |
|---|---|---|
| Abacus-212 (`datasets/abacus229.csv`) | ChatGPT web-UI conversations, 49 calculators | Free text only, no per-field labels |
| Abacus-409 (`datasets/abacus409.csv`) | The **409 incorrect** answers out of 10,000 trials, with error classes | Failures only, so there is no denominator; no per-field labels |

### 2.4 Why extraction deserves its own benchmark

Error classes in Abacus-409 (n = 409):

| Error class | n | % |
|---|---|---|
| Interpretation Error | 142 | 34.7 |
| Assignment Error | 89 | 21.8 |
| Incorrect Formula | 80 | 19.6 |
| Incorrect Criteria | 41 | 10.0 |
| Calculation Error | 27 | 6.6 |
| Software Error/Other | 15 | 3.7 |
| Indeterminate | 10 | 2.4 |
| Incorrect Reporting | 5 | 1.2 |

Interpretation and Assignment errors make up **56.5%** of observed failures. These are errors in turning text into calculator inputs. A typed API cannot fix them. Only better extraction can.

---

## 3. Research questions

- **RQ1. Accuracy by model family.** When a deterministic API computes the score, how accurately do the model families fill the typed request? Measured per field, per case, and on the final score.
- **RQ2. Decomposition versus agents.** Does "extract, then compute" match or beat the paper's end-to-end arms, including the agentic OMC arm, on the same cases?
- **RQ3. Schema design.** Does accuracy change when the schema asks for *decisions* (`hypertension: bool`, meaning uncontrolled with SBP > 160) rather than *raw facts* (`systolic_bp: 165`, with the API applying the threshold)?
- **RQ4. Robustness.** How does performance hold up under paraphrase, distractors, conflicting values, unit changes, longer notes, and real clinical text?
- **RQ5. Cost and deployment.** What are the cost, latency, and on-premises feasibility trade-offs? This matters because clinical text may contain protected health information (PHI).

### Pre-registered hypotheses

- **H1.** On the in-distribution templated vignettes, rules and fine-tuned encoders will be near ceiling, because each criterion has very few fixed phrasings (see §5.3).
- **H2.** On paraphrased and perturbed text, schema-constrained LLMs and NuExtract will beat rules and encoders. The gap will be largest for negation scope, laterality (NIHSS), and multi-select criteria (Caprini).
- **H3.** "Extract, then compute" with a constrained LLM will have fewer final-score errors than the same LLM's base, RAG, and code-interpreter arms. It will match or beat its OMC arm.
- **H4.** Raw-facts schemas will improve encoder and rule accuracy on threshold-bearing fields more than they improve LLM accuracy.

---

## 4. Study design

### 4.1 The pipeline

```
clinical text ──► extractor ──► typed request (Pydantic, validated) ──► OpenMedCalc ──► score
                     ▲                         │
                     │                         └── graded per field against the gold request
               one system per arm                                       │
                                                score graded against the gold score
```

Every system uses the same interface:

```python
def extract(text: str, schema: type[BaseModel]) -> BaseModel: ...
```

No parser LLM is used. Outputs must validate against the schema. A validation failure counts as an error for every field, and it is also reported on its own.

### 4.2 Calculators and schemas

The ten calculators from the paper's main experiment are in scope. The typed request models already exist in `llm_calc/tools/openmedcalc/api_datamodel.py`.

| Calculator | Request model | Schema fields | Field types | Vignette criteria | Phrase options |
|---|---|---|---|---|---|
| ARISCAT | `CalcRequestAriscat` | 7 | int, float, bool, enum | 7 | 16 |
| Caprini VTE | `CalcRequestCapriniVte` | 32 | int, bool, enum | 8 | 46 |
| Charlson (CCI) | `CalcRequestCCI` | 17 | int, bool, enum | 17 | 40 |
| GAD-7 | `CalcRequestGAD7` | 8 | enum (ordinal) | 8 | 31 |
| HAS-BLED | `CalcRequestHASBLED` | 9 | int, bool | 9 | 24 |
| MELD-Na | `CalcRequestMeldNa` | 5 | float, bool | 5 | 19 |
| NIHSS | `CalcRequestNIHSS` | 15 | enum (ordinal, lateralised) | 15 | 56 |
| PSI/PORT | `CalcRequestPsiPort` | 20 | int, float, bool, enum | 20 | 41 |
| SOFA | `CalcRequestSOFA` | 13 | int, float, bool | 9 | 36 |
| Wells DVT | `CalcRequestWellsDvt` | 10 | bool | 10 | 23 |

Where the number of vignette criteria differs from the number of schema fields, the mapping is not one-to-one:

- **Caprini:** multi-select criteria such as `recent_history` and `venous_disease` fan out into many booleans.
- **SOFA:** several fields come from a single criterion.

The mapper in §5.1 must handle both cases explicitly.

### 4.3 Schema variants (for RQ3)

- **Variant D: decision schema.** The existing OpenMedCalc request models, unchanged.
- **Variant R: raw-facts schema.** Threshold-bearing fields are replaced by the underlying measurement, and the threshold logic moves into the API. Examples:

  | Decision field (Variant D) | Raw-fact fields (Variant R) |
  |---|---|
  | `hypertension: bool` | `systolic_bp: int \| None` |
  | `severe_renal_disease: bool` | `creatinine_mg_dl: float \| None`, `on_dialysis: bool \| None`, `kidney_transplant: bool \| None` |
  | `alcohol: bool` | `alcoholic_drinks_per_week: float \| None` |
  | `age: int` (banded inside the calculator) | `age_years: int \| None` |

  Variant R is built only where a threshold exists. All other fields are the same in both variants.

  *As implemented:* most OpenMedCalc request models already take raw values. MELD-Na, SOFA and PSI/PORT take labs and vitals; ARISCAT and CCI take ages and SpO₂. The threshold-in-boolean pattern is concentrated in HAS-BLED, so Variant R exists for HAS-BLED (`schemas_raw.RawRequestHASBLED`, whose `to_decision()` applies the thresholds). Wells DVT and Caprini fold clinical judgement into booleans and enums rather than numeric thresholds, so a raw-facts variant would need a different design; that is left open.

### 4.4 Missing and negated information

Clinical extraction has to tell apart:

- **asserted present**
- **asserted absent** (negated)
- **not documented**

The existing schemas use plain `bool`, which merges the second and third. For this benchmark:

- Every field becomes `Optional`, with `None` meaning "not documented".
- The gold labels separate "absent" from "not documented" based on what the vignette actually states. Several templates say "has none of the following: …", which counts as documented absence. Some cases say "unknown age".
- For scoring, the API defaults "not documented" to the calculator's convention (usually 0 points). This is reported as its own metric (§8.3).

*As implemented:* Variant D keeps the OpenMedCalc request models unchanged, so every field is required. Extractors are told to answer `false` (or a 0 dose) for criteria the text does not mention. Variant R fields are `Optional`. In the templated sets every criterion is stated or covered by a template sentence such as "no other venous or clotting disorders", so gold "not documented" occurs only in Variant R (dialysis and transplant in HAS-BLED). §8.3 becomes informative once T3, T5 or T6 provide genuinely undocumented fields.

---

## 5. Data

### 5.1 Phase 0: per-field gold labels

**Gap:** the released datasets have no per-field labels. The generator does have them, because every case stores the sampled `CriterionOption` list (`Case.options`).

Tasks:

1. **Persist options.** Extend case generation to write, for each case: the vignette, the sampled options, the **gold request** under Variant D (and Variant R), the gold score, and the template ID.
2. **Mapper.** Write `options → CalcRequestX` for each calculator. It is hand-written and reviewed by a clinician, because it encodes clinical meaning (for example, which `recent_history` phrase sets which Caprini boolean).
3. **Round-trip check.** For every generated case, `OpenMedCalc(gold_request)` must equal `correct_output` from `calculate_output_for_case`. Any mismatch is a mapper bug, or a bug in the original ground truth, and is logged and fixed before any model is evaluated. This check is also a useful audit of the original study's ground truth.
4. **Reproducibility.** Fixed seeds, as in the existing generator. The generated datasets are versioned and released.

### 5.2 Test sets

| Set | Description | Purpose | Target size |
|---|---|---|---|
| **T1: in-distribution** | Generated exactly as in the paper | Comparability with the original arms | 100 cases × 10 calculators |
| **T2: held-out phrasing** | New phrasings for every criterion option, unseen in any training data | Generalisation beyond memorised strings | 100 × 10 |
| **T3: full paraphrase** | Whole vignettes rewritten (LLM paraphrase, then clinician check of a stratified sample of at least 20%) | Realistic surface variation | 100 × 10 |
| **T4: stress** | Controlled perturbations, listed below | Failure-mode analysis | 50 × 10 per perturbation |
| **T5: external** | MedCalc-Bench (Khandekar et al., 2024), restricted to overlapping calculators, using its parameter annotations | External validity on case-report text | All overlapping instances |
| **T6: real notes** *(optional)* | De-identified notes, e.g. MIMIC-IV under its data use agreement, annotated by two clinicians | Clinical validity | 30–50 per calculator |

**T4 perturbations (as implemented in `perturb.py`; 1,869 cases from the first 50 T1 cases per calculator):**

- *Distractors (452):* irrelevant sentences, including a family-history sentence that names scored conditions ("their father had a stroke…"), which tests attribution.
- *Conflicting values (150):* an older, different value next to each current lab, e.g. "creatinine 1.3 (it was 2.4 three days ago)".
- *Units (200):* µmol/L instead of mg/dL, °F instead of °C, mmol/L glucose. Values are written with just enough precision to convert back within 0.5%.
- *Abbreviations and shorthand (332):* e.g. "h/o DVT/PE", "EtOH 9 drinks/wk", "On HD TIW", "PMHx significant for CVA".
- *Reorder (250):* the option sentences in random order. This defeats position cues and applies to list-style templates only.
- *Long note (485):* the vignette embedded in a note with unrelated sections.

Not yet implemented: laterality swaps (NIHSS left/right is covered by its own fields) and dedicated negation-scope rewrites (partly covered by the abbreviation and distractor sets).

**T5 notes:** check MedCalc-Bench's licence and which calculators it covers before committing to this set. Its parameter names must be mapped to our schemas with the same round-trip check as §5.1.

*As implemented (`medcalc.py`):*
- **Source:** MedCalc-Bench Verified at a pinned revision, CC-BY-SA-4.0. T5 is built on demand and never committed.
- **Coverage:** seven calculators overlap (Caprini, CCI, HAS-BLED, MELD-Na, PSI/PORT, SOFA, Wells DVT), giving 654 rows from both splits.
- **Validity:** 637 rows map to valid requests. The other 17 have values outside the OpenMedCalc ranges.
- **Agreement:** OpenMedCalc's score matches MedCalc-Bench's answer in 629 of 637 cases.
- **Undocumented criteria:** criteria the annotators did not record are kept as "not documented" (21% of field instances), which makes T5 the set for §8.3.

**T6 notes:** real notes must never be sent to external APIs unless the data use agreement explicitly allows it. On T6, run only on-premises systems (rules, encoders, NuExtract, open-weight LLMs).

### 5.3 Leakage and memorisation risk

The generator uses **one fixed phrase per option**, and most criteria have only two options (positive and negative). For example, HAS-BLED has 24 phrases in total. A model fine-tuned on T1-style data can reach near-perfect accuracy by matching strings, without understanding the text.

Mitigations:

- Training data for fine-tuned systems (§6) comes **only** from a separate pool of phrasings. T2 and T3 phrasings never appear in training.
- Split by template and phrasing, never just by case.
- Report T1 results for comparability only. **T2 and T3 are the primary evaluation sets.**

*Confirmed in Phase 0:* a template-inversion baseline, which knows the templates and option phrasings and nothing clinical, scores **100%** on T1 and on all 397 matched Abacus-409 failures. It falls to 6.3% on T4. T1-style text is fully solvable by string matching.

---

## 6. Systems under comparison

All systems output the same typed request. Fine-tuned systems are trained on a **synthetic training pool**: the generator run with a disjoint phrasing pool, about 1,000–5,000 cases per calculator.

### A. Rule-based clinical NLP

- medspaCy with ConText (negation, history, hypotheticals), regular expressions for numbers and units, and the target-rule matcher for criteria.
- Optionally, QuickUMLS or scispaCy for concept normalisation.
- Rules are written from the **training** phrasing pool only.
- **Role:** a transparent baseline that shows how much of the benchmark plain pattern matching can solve.

### B. Encoder models (BERT family)

Candidates: BioClinicalBERT, PubMedBERT, GatorTron, and a modern long-context encoder (e.g. a ModernBERT clinical or biomedical variant) for the long-note perturbation.

Formulations by field type:

| Field type | Formulation |
|---|---|
| bool / tri-state / enum / ordinal | Multi-head sequence classification: one head per schema field over the whole vignette |
| Numeric | Extractive QA per field (`"What is the serum creatinine?"`), then deterministic unit parsing |
| Multi-select (Caprini) | Multi-label head |

Also include a zero-shot natural language inference (NLI) baseline for boolean fields: a DeBERTa-NLI model scoring the premise "vignette" against the hypothesis "The patient has cirrhosis."

### C. Dedicated small extraction models

- **NuExtract** (NuMind; versions 1.5 and 2.0, several sizes). Each Pydantic model is converted to a NuExtract JSON template, with enums as option lists and booleans as `["true", "false", "not_documented"]`. Run zero-shot first, then fine-tuned on the synthetic training pool.
- **GLiNER2 / GLiNER-biomed**, zero-shot, for span-style fields.

### D. LLMs with schema-constrained decoding

- **Paper models:** GPT-4o and Llama 3.1, for direct comparability with the original arms.
- **Current frontier closed models**, from at least two providers.
- **Open-weight models of several sizes**, served with guided decoding (vLLM, Outlines, or XGrammar).
- **Settings:** strict JSON-schema mode, temperature 0, the same instruction prompt for every model, and field descriptions copied verbatim from the Pydantic schema.
- **Prompt variants:** zero-shot, and few-shot (3 examples from the training pool).

### E. Reference arms from the original study

The original `base`, `rag`, `ci`, `rag_ci`, and `omc` arms are re-run on T1–T3 with the same cases, so the end-to-end and "extract, then compute" approaches can be compared directly (RQ2).

---

## 7. Experimental matrix

| Factor | Levels |
|---|---|
| System | A, B (each encoder), C (each model and size, zero-shot or fine-tuned), D (each LLM, zero-shot or few-shot), E (each arm) |
| Schema | Variant D, Variant R (for systems A–D) |
| Test set | T1, T2, T3, T4 (each perturbation), T5, T6 |
| Calculator | 10 |
| Repeats | 1 for deterministic systems; 3 for API LLMs, to measure run-to-run variation even at temperature 0 |

To control cost:

- System E runs on T1–T3 only.
- Frontier LLMs run on T4 and T5 only after a pilot on a 20% subsample confirms the pipeline works.

---

## 8. Outcomes and metrics

### 8.1 Primary endpoints

Both are computed on T2 ∪ T3:

1. **Final-score exact accuracy:** the share of cases where `OpenMedCalc(predicted_request) == gold_score`. This is directly comparable with the paper.
2. **Macro field accuracy:** per-field accuracy, averaged within each calculator and then across calculators. A field is correct when it matches the gold value exactly, or for floats within ±1% (unit conversions round). Exact float match is reported as a secondary column.

### 8.2 Secondary endpoints

- **All-fields-correct rate:** the share of cases where every field is right. Final-score accuracy is never lower than this; the difference measures how often errors cancel out.
- **Risk-category agreement:** agreement on clinically used cut-offs, which matters more at the bedside than exact points. Examples: HAS-BLED ≥ 3, Wells DVT likely or unlikely, PSI class I–V, Caprini risk tier, NIHSS severity band, GAD-7 severity band.
- **Score absolute error**, and its signed direction (systematic over- or under-scoring).
- **Accuracy by field type:** boolean, ordinal enum, nominal enum, integer, float (exact, and within ±1% after unit normalisation), multi-select (micro-F1).
- **Schema validity rate:** the share of outputs that pass validation without repair.

### 8.3 Assertion handling

For tri-state fields, report a confusion matrix of present / absent / not documented. Two errors are reported separately because they have different clinical consequences:

- **hallucinated presence:** predicting *present* when the field is not documented;
- **missed negation:** predicting *present* when the field is asserted absent.

### 8.4 Operational metrics

Per 1,000 cases, report:

- median and p95 latency;
- cost (API price, or GPU-hours for local systems);
- peak memory;
- whether the system can run fully on-premises.

---

## 9. Statistical analysis

- **Unit of analysis:** the case for case-level endpoints; the case × field pair for field-level endpoints.
- **Confidence intervals:** 95% CIs from a case-level bootstrap with 2,000 resamples, clustered by case for field-level metrics.
- **Paired comparisons:** all systems see identical cases.
  - McNemar's test for binary case-level outcomes.
  - The paired bootstrap for differences in macro accuracy.
  - Holm–Bonferroni correction within each pre-specified family of comparisons. The families are:
    - (i) the best system of each family against each other on the primary endpoints;
    - (ii) Variant D against Variant R within each system;
    - (iii) extract-then-compute against the OMC arm, per LLM.
- **Modelling:** mixed-effects logistic regression on field-level correctness, with random intercepts for calculator, case, and field:
  - `correct ~ system + schema_variant + field_type + test_set + system:test_set + (1|calculator) + (1|case) + (1|field)`
  - The `system:test_set` interaction is the formal test of H1 and H2 (robustness degradation).
- **Sample size:** at an expected accuracy of 0.90, 1,000 cases per test set give a 95% CI half-width of about ±1.9 percentage points overall. Per calculator (n = 100), the half-width is about ±5.9 points. Per-calculator results are therefore descriptive. Pooled results carry the inferential claims.
- **Run-to-run variation:** for LLMs with 3 repeats, report the mean and range, and the share of cases whose correctness flips between repeats.

---

## 10. Error analysis

1. **Map to the original taxonomy.** Classify every final-score error using the Abacus-409 classes. By construction, extract-then-compute systems can only produce Interpretation and Assignment errors (plus Software). Confirming this empirically validates the decomposition.
2. **Extraction-specific taxonomy**, applied at field level:

   | Code | Description |
   |---|---|
   | `MISS` | Fact present in text, not extracted |
   | `HALLUC` | Value extracted that is not supported by the text |
   | `NEG` | Negation or assertion error |
   | `TEMP` | Wrong time point (historical vs. current; old vs. new lab) |
   | `UNIT` | Unit or conversion error |
   | `THRESH` | Raw fact right, threshold decision wrong (Variant D only) |
   | `LAT` | Laterality error (NIHSS) |
   | `ORD` | Off-by-one on an ordinal scale |
   | `MULTI` | Multi-select error (missed or extra item) |
   | `SCHEMA` | Validation failure |

3. **Annotation process.** Two annotators label a stratified sample of 300 field-level errors. Report Cohen's κ and resolve disagreements by adjudication.

   *As implemented* (`python -m llm_calc.extraction annotate`):
   - **`sample`** draws errors balanced over system family × calculator and writes one blinded TSV sheet per annotator. System names are hidden behind a uid, with the key kept in a separate file.
   - **`agree`** reports raw agreement and Cohen's κ, overall and per code, and writes `adjudicate.tsv` listing every disagreement.
   - **`count`** turns the agreed and adjudicated sheets into `error_codes.tsv`, which feeds Figure 10 and Table 8.
   - **A `GOLD` code** lets annotators flag a suspected gold-label error instead of a model error.
4. **Clinical impact.** For each error, record whether it changes the risk category (§8.2). Report the share of errors that change management-relevant categories.

---

## 11. Implementation plan

### 11.1 Repository layout (as built)

```
llm_calc/extraction/
├── __main__.py        # CLI: python -m llm_calc.extraction <command>
├── omc.py             # OpenMedCalc request models + calculators, loaded without the web app
├── generator.py       # exact re-implementation of the original vignette generator
├── gold.py            # options → CalcRequest mappers (Variant D), round-trip check
├── schemas_raw.py     # Variant R (HAS-BLED) and its threshold logic
├── schemas.py         # field specs, JSON schemas, prompts
├── datasets.py        # T1, TRAIN, A409, T4 builders; JSONL I/O
├── perturb.py         # T4 perturbations
├── phrasings.py       # T2/T3: option export, review sheets, set builders
├── paraphrase.py      # GPU job: paraphrase candidates for clinician review
├── units.py           # number parsing (incl. number words) and unit normalisation
├── evaluate.py        # field- and case-level grading, risk bands, summaries
├── stats.py           # Wilson CI, exact McNemar, bootstrap, Holm
├── attribution.py     # Abacus-409 error attribution with a mirrored-error control
└── extractors/
    ├── base.py        # Extractor interface, JSON parsing
    ├── baselines.py   # oracle, majority, template inversion
    ├── rules.py       # System A
    ├── encoder.py     # System B: shared encoder, classification + span heads
    └── llm.py         # Systems C/D: NuExtract and chat LLMs via vLLM; OpenAI-compatible APIs
cluster/killarney/     # Slurm jobs, environment setup, model manifest (§11.4)
tests/test_extraction.py
```

- The CLI is a standalone module rather than an `llmcalc` subcommand. The existing CLI builds LLM clients and needs API keys at import time; the benchmark must run on offline compute nodes without them.
- Results are JSONL and TSV files (`predict` writes one file per system, set and variant; `evaluate` writes the summaries) rather than DuckDB tables. Flat files are easier to move between scratch and project space, and to diff.
- GLiNER is not implemented yet.

### 11.2 Reproducibility

- Pin model versions and record them with each run, along with prompts, decoding parameters, and seeds.
- Store raw model outputs before validation.
- Release the generated T1–T4 sets and gold requests as **Abacus-Extract**, under the same terms as the existing datasets.

### 11.3 Milestones

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| 0 | Gold-label generator and mappers for 10 calculators | 100% round-trip agreement on 1,000 generated cases (or every discrepancy explained) | **Done:** 10,000/10,000; A409 397/397 matched |
| 1 | T1–T3 test sets and the training phrasing pool | Clinician sign-off on the T2 phrasings and the T3 paraphrase sample | T1, T4, TRAIN done; T2/T3 wait on the paraphrase job and review |
| 2 | Extractors A–D implemented, pilot on 20% of T2 | All systems produce schema-valid outputs end to end | Implemented; CPU and stub dry runs pass; GPU pilot on Killarney next |
| 3 | Full runs on T1–T3, plus E arms | Primary endpoints computed | Waiting on Killarney (`submit.sh all`) |
| 4 | T4 stress, T5 external, optional T6 | Robustness analysis complete | T4 built; T5/T6 not started |
| 5 | Error annotation, statistics, write-up | Manuscript draft and dataset release | Not started |

### 11.4 Compute: Killarney (Alliance / PAICE)

All GPU work runs on Killarney under the group's `aip-` allocation. The scripts and instructions are in [`cluster/killarney/`](../cluster/killarney/README.md). Design choices that follow the Alliance documentation:

- **Hardware:** standard nodes have 4 × L40S 48 GB, 64 cores and 512 GB RAM. Each GPU is requested with 12 cores and 100 GB.
  - Models up to about 9B parameters run on 1 GPU.
  - 32B models run on 2 GPUs (tensor parallel).
  - 70–72B models in bf16 run on 4 GPUs of one node, so no multi-node Ray is needed.
  - The H100 nodes (8 × 80 GB) are an alternative for the 70B rows (`gpu` column of `models.tsv`).
- **Offline compute:** every model is downloaded on a login node (`download_models.sh`). Jobs run with `HF_HUB_OFFLINE=1`.
- **Software:** virtualenvs built from Alliance wheels (`--no-index`): an `extract` env (python/3.11, torch, transformers) and a `vllm` env (python/3.13 + opencv, as the Alliance vLLM page requires).
- **Storage:**
  - Predictions go to `$SCRATCH` (purged after 60 days).
  - `evaluate` copies summaries and a tarball of predictions to project space.
  - Models live in project space by default (70B ≈ 140 GB each).
- **Submission:** `submit.sh` reads `models.tsv`, submits one job per row with matching `--gpus-per-node`, `--time` and `--mem`, makes GPU jobs wait on the CPU `prepare` job, and runs `evaluate` once all of them finish. It refuses to run from `$HOME`, which Killarney does not allow.
- **Requested budget:** the upper bound from walltime limits is about 170 GPU-hours. Actual use should be well below that, since walltimes are generous.

  | Stage | GPU-hours |
  |---|---|
  | 4 encoders | 24 |
  | 6 NuExtract configurations | 17 |
  | 7 LLM configurations (two at 70B) | 115 |
  | Paraphrase jobs | 16 |

- **Data governance:** synthetic sets may go to external APIs (the GPT-4o arm). Real notes (T6) are processed only on Killarney.



---

## 12. Risks and limitations

- **Synthetic text.** Even paraphrased vignettes are cleaner than real notes. T5 and T6 reduce this risk, but conclusions about real deployment should rest mainly on them.
- **Paraphrase drift.** LLM paraphrasing can change clinical meaning. The clinician check of a sample (§5.2) measures how often; flagged cases are corrected or dropped, and the drift rate is reported.
- **Paraphrase-model bias.** If the paraphrasing LLM also appears as a system under test, it may have an advantage. Paraphrase with a model family that is not evaluated, or with several families.
- **Schema choices shape the results.** RQ3 tests this directly. Any final claim should name the schema variant it applies to.
- **Gold-label fidelity.** Gold labels come from the generator's option metadata, so the round-trip check (§5.1) and clinician review of the mappers are essential.
- **Model churn.** API model versions change. Pin versions, record dates, and present frontier-model results as a snapshot.

---

## 13. Open decisions

1. Is T6 (real notes) in scope, and under which data use agreement?
2. Should the calculators expand beyond the 10 to cover more of the 49 from Abacus-212? This needs new typed request models for each added calculator.
3. Which risk cut-offs are pre-specified for each calculator? Clinical input is needed.
4. What is the compute budget for fine-tuning (encoders and NuExtract) and for frontier-API runs? Killarney covers the GPU side (§11.4). API runs need keys and a budget.
5. Clinician adjudication of the CCI liver-disease option in the released data: cirrhosis "awaiting transplant evaluation" was scored as moderate/severe (3 points), but the text only supports mild (see the results report §2.3).
6. Killarney: which `aip-` allocation to use, whether project quota allows the 70B models in project space, and an HF token with the Llama 3.1 licence accepted.

---

## Appendix A. Example gold record (HAS-BLED, Variant D)

```json
{
  "case_id": 1042,
  "calculator": "hasbled",
  "template_id": 2,
  "test_set": "T1",
  "vignette": "I need your help. I have a patient. They are a 71-year-old woman. They have a history of hypertension and are not on treatment. They clinic blood pressure is 165/85. ...",
  "gold_request": {
    "age": 71,
    "hypertension": true,
    "severe_renal_disease": false,
    "severe_liver_disease": false,
    "stroke_history": false,
    "prior_major_bleeding": true,
    "labile_inr": false,
    "drugs": true,
    "alcohol": false
  },
  "gold_assertions": { "alcohol": "absent", "labile_inr": "absent" },
  "gold_score": 4
}
```

## Appendix B. Example NuExtract template (HAS-BLED, Variant R)

```json
{
  "age_years": "integer",
  "systolic_bp": "integer",
  "on_antihypertensive": ["true", "false", "not_documented"],
  "creatinine_mg_dl": "number",
  "on_dialysis": ["true", "false", "not_documented"],
  "cirrhosis": ["true", "false", "not_documented"],
  "stroke_history": ["true", "false", "not_documented"],
  "prior_major_bleeding": ["true", "false", "not_documented"],
  "labile_inr": ["true", "false", "not_documented"],
  "antiplatelet_or_nsaid_use": ["true", "false", "not_documented"],
  "alcoholic_drinks_per_week": "number"
}
```
