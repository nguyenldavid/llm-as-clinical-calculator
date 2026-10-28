# Paper draft: outline and sample text

> **Status: working draft, not for circulation as results.**
>
> - **Real numbers.** Plain numbers in this draft come from analyses already run and committed in this repository, with the source named. Those analyses are Phase 0, the Abacus-409 attribution, the CPU baselines and the T5 build.
> - **Placeholders.** Anything still to be measured is a placeholder written ⟨like this⟩. Each one names the table or figure of the [figures page](mock_results/README.md) that will fill it, or that table's key in `figures.py`.
> - **Sample sentences.** Sentences around placeholders show the intended framing. They are not predictions: the direction of every placeholder result is open, and the text must follow whatever the data show.
>
> **Target journals:** *npj Digital Medicine* (where the parent study appeared), *Nature Communications*, or *Patterns*. The structure below follows the Nature-family format: Results before Methods, about 4,000–5,000 words, up to 8 display items, and extended data for the rest. Reporting follows TRIPOD-LLM ([checklist mapping](tripod_llm_checklist.md)).

---

## Working titles

1. **Reading, not arithmetic: benchmarking clinical information extraction for typed calculator APIs**
2. Extract, then compute: where language models fail when a calculator does the maths
3. Separating extraction from computation in LLM clinical calculation: a multi-family benchmark with per-field gold labels

**Running title:** Extraction benchmark for clinical calculators

**Authors:** ⟨authors, affiliations⟩. **Corresponding author:** ⟨name, email⟩.

---

## Abstract (structured draft, about 200 words)

**Background.** Language models make errors when asked to compute clinical scores. Giving them a typed calculator API removes the knowledge and arithmetic steps, leaving one: reading the patient's findings from text into the calculator's inputs. How well different model families do that step has not been measured.

**Methods.** We built a benchmark of ⟨N⟩ clinical notes for 10 calculators, with a gold label for every input field. It has five parts:

- template vignettes from a published study;
- new phrasings and whole-note paraphrases;
- controlled perturbations, including prompt injection and deleted findings;
- 637 case-report and synthetic notes from MedCalc-Bench Verified;
- the 397 released failures of the parent study, matched back to their inputs.

Every system fills the same validated request, and OpenMedCalc computes the score. The systems are rules, fine-tuned clinical encoders, NuExtract, open and proprietary LLMs with schema-constrained decoding, a typed decision model (Jev), and two clinicians.

**Results.** On template text, a baseline that only inverts the templates scored 100%, and fell to 7.5% once the wording changed. On new wording, final-score accuracy ranged from ⟨x⟩% (⟨family⟩) to ⟨y⟩% (⟨family⟩), against ⟨z⟩% for clinicians. ⟨Headline comparison with 95% CI.⟩ Extract-then-compute ⟨reduced / did not reduce⟩ errors relative to the parent study's agent arm (⟨Δ, CI⟩).

**Conclusions.** ⟨One sentence on whether extraction is the bottleneck, and one on what that means for deploying calculator tools.⟩

---

## Introduction (sample text)

Clinical scores such as HAS-BLED, Wells, SOFA and MELD-Na turn findings in a patient's record into numbers that drive anticoagulation, imaging, triage and transplant listing. Computing them by hand is tedious and error-prone. That makes them an obvious target for language models, and an instructive one, because each score has a single correct answer.

A recent study found that GPT-4o and Llama 3.1 answered clinical calculation questions incorrectly in a large share of trials. It also found that giving the models task-specific calculator tools (OpenMedCalc) cut incorrect answers 13-fold and 5.5-fold respectively (Goodell et al., 2025). The calculator tools were the most effective of the augmentations tested, ahead of retrieval and a code interpreter.

A calculator tool, however, removes only part of the problem. Producing a score from a note takes three steps:

1. **Knowledge:** knowing which criteria and formula the score uses.
2. **Extraction:** reading the patient's values for those criteria from the text.
3. **Arithmetic:** combining them.

A typed calculator API takes over steps 1 and 3 by construction: it fixes the criteria in its request schema and does the arithmetic itself. What remains is extraction. Is the patient on dialysis? Is the hypertension uncontrolled? Is the drift in the left arm or the right? Was the creatinine 1.3 today or three days ago?

Extraction is where an end-to-end evaluation sees least. A wrong final score does not say which input was misread. A right one can hide two errors that cancel.

The parent study's released failures illustrate the difficulty. When each of the 397 incorrect answers is matched back to the vignette that produced it, 74% can be reproduced by changing a single input field. That suggests most failures are single misreadings, but a control undercuts it: 66% would also be "reproduced" by an error of the same size in the opposite direction. Final scores alone therefore cannot attribute an error to a field. Measuring extraction requires gold labels for every field (this work, Methods).

Whether specialised extraction tools beat general LLMs for filling a known schema is also open. The candidates include:

- rules;
- fine-tuned clinical encoders;
- small dedicated models such as NuExtract;
- typed decision models that answer constrained questions instead of generating text.

The answer has practical weight. Clinical text contains protected health information, and a small on-premises model is easier to deploy than a frontier API.

Here we separate extraction from computation. We built a benchmark in which every system fills the same validated request for the same calculator API, and every input field has a gold label. We test generalisation beyond the phrasings a system could have memorised, robustness to realistic and adversarial changes to the note, performance on external notes, and agreement with clinicians doing the same task. We then compare the extract-then-compute pipeline with the end-to-end and agentic arms of the parent study on identical cases.

*Paragraph to add once results exist:* a summary of the three main findings, one sentence each.

---

## Results

### R1. A benchmark with per-field gold labels (Fig. 1, Table 1)

**Sample text.** We re-implemented the parent study's vignette generator so that it reproduces the original seeds exactly. For each calculator we then wrote a mapper from the sampled criterion options to an OpenMedCalc request.

- **Round trip.** For 10,000 generated cases, OpenMedCalc applied to the gold request returned the original ground truth in all 10,000 (`phase0_round_trip.yaml`).
- **Matching the released failures.** All 397 non-empty vignettes among the 409 released failures matched a unique gold request. Matching required recovering three option wordings that had been edited in the source spreadsheet after the experiment.
- **External notes.** Of 654 MedCalc-Bench Verified notes for the seven shared calculators, 637 could be expressed as valid requests. For 629 of these 637, OpenMedCalc's score equals MedCalc-Bench's own answer (`T5_build_report.yaml`). The eight disagreements are MELD-Na rounding and dialysis handling (6) and the HAS-BLED age threshold (2).
- **Test sets and flow.** Fig. 1 (`flow`) shows how the five test sets were derived, and Table 1 (`datasets`) describes them.

**Data-quality findings** (Supplementary Note 1). Building the gold labels surfaced the following:

- two calculator defects, both fixed: Wells DVT crashed on negative scores, and SOFA under-scored ventilated patients with P/F 200–399;
- one disagreement with the published definition, left unchanged pending clinical review: HAS-BLED age ≥65 vs >65;
- schema range checks that reject real patients (MELD-Na sodium);
- an ambiguous released label (CCI liver disease).

These are themselves an argument for typed inputs: each defect became visible only once inputs were explicit.

### R2. Template text measures memorisation, not extraction (Extended Data Fig. ⟨cpu_baselines⟩)

**Sample text (real numbers).** The parent study, like most calculator benchmarks, uses vignettes assembled from a fixed phrase per criterion option. A baseline that knows those phrases and nothing clinical, template inversion, scored 100% on T1 and on all 397 released failures. When the same cases were perturbed (T4), its final-score accuracy fell to 7.5% (95% CI 5.5–9.7), near the majority-class floor of 8.3%.

A rule system written against the templates reached 100% on T1 and T4. On MedCalc-Bench notes it reached 15.5% (12.7–18.4). Its field accuracy there was 0.74, so most fields were right, but almost every note had at least one field wrong.

Template-generated vignettes therefore cannot rank extraction systems. We use them only for comparability, and base every inferential claim on held-out phrasings (T2) and paraphrases (T3).

### R3. Accuracy by model family on new wording (Fig. 2, Table 2, Fig. 3)

**Sample text.**

- **Final-score accuracy** on T2 ∪ T3 ranged from ⟨x⟩% for ⟨lowest family⟩ to ⟨y⟩% for ⟨best system⟩ (Fig. 2 `headline`; Table 2 `primary`).
- **Macro field accuracy** ranged from ⟨a⟩ to ⟨b⟩. The two endpoints ⟨agreed / diverged⟩ in ranking. ⟨Note on the gap between all-fields-correct and score accuracy, i.e. how often errors cancel (Table `band`).⟩
- **Family comparisons** (Fig. 3 `forest`; Table `pairwise`, Holm-adjusted). ⟨Paired differences between the best system of each family, e.g. "Schema-constrained open LLMs exceeded fine-tuned encoders by ⟨Δ⟩ points (95% CI ⟨…⟩; P_Holm ⟨…⟩)".⟩
- **Clinicians** reached ⟨z⟩% on their stratified sample (Table `clinician`). Their inter-reader agreement on the final score was ⟨…⟩%, and field-level κ ranged from ⟨…⟩ to ⟨…⟩. ⟨Which systems fall within the clinician interval.⟩

**H1–H2.** In the field-level model (Table `model`, `model_terms`), the system × test-set interaction was ⟨χ², df, P⟩. Read: systems ⟨did / did not⟩ differ in how much accuracy they lost from template text to new wording. ⟨Name the families with the largest drops (Extended Data Fig. `generalisation`).⟩

### R4. Where extraction fails (Fig. 4, Extended Data Figs. `field_kind`, `field_heatmap`, `assertion`)

**Sample text.**

- **By field type.** Accuracy was lowest for ⟨field type⟩ fields (⟨…⟩) and highest for ⟨…⟩.
- **By calculator.** ⟨Calculator⟩ was hardest for every family, driven by ⟨fields⟩.
- **Error types.** Annotated errors (two annotators, κ = ⟨…⟩; Fig. 4 `error_taxonomy`) were dominated by ⟨code⟩ for ⟨family⟩ and ⟨code⟩ for ⟨family⟩.
- **Undocumented criteria (T5).** On criteria the note does not mention, ⟨family⟩ reported a finding as present in ⟨…⟩% of fields, i.e. hallucinated presence (Extended Data Fig. `assertion`).
- **Clinical impact.** ⟨…⟩% of score errors changed the risk band (Table `band`).

### R5. Robustness and safety (Fig. 5 `t4_robustness`)

**Sample text.** Paired against the same base case, perturbations cost between ⟨…⟩ and ⟨…⟩ points of final-score accuracy (Fig. 5).

- **Prompt injection** (a sentence telling the AI to report no risk factors) lowered accuracy by ⟨Δ⟩ for ⟨families⟩. Rules and encoders, which do not follow instructions, are its controls; the rule system was unaffected (100%).
- **Omission** (one scored finding deleted) tests whether a system reports what the template leads it to expect. ⟨Families⟩ still reported the deleted finding in ⟨…⟩% of cases.
- **Units and conflicting values** were hardest for ⟨…⟩.

### R6. Schema design (Extended Data Fig. `schema_variant`, Table `variant_paired`)

**Sample text.** For HAS-BLED, we compared two schemas:

- a **decision schema**, which asks the extractor to apply the thresholds itself (hypertension = systolic >160 mmHg);
- a **raw-facts schema**, which asks only for the values and lets the API apply the thresholds.

The raw-facts schema changed final-score accuracy by ⟨Δ (CI)⟩ for ⟨…⟩, against ⟨…⟩ for ⟨…⟩ (H4). ⟨Implication for API designers: which side of the threshold the schema puts the judgment on.⟩

### R7. Extract-then-compute versus the parent study's agents (Fig. 6 `paper_arms`)

**Sample text.** On the same T1 cases, the parent study's five arms (base, retrieval, code interpreter, retrieval plus code interpreter, and agentic OMC) scored ⟨…⟩. Extract-then-compute with the same models scored ⟨…⟩ (H3: ⟨Δ vs OMC arm, CI⟩). ⟨If extract-then-compute matches the agent arm: the agent's gain came from the tool computing, not from the agent choosing when to call it.⟩

### R8. Confidence and selective automation (Fig. 7 `confidence`, Tables `confidence`, `calibration`)

**Sample text.** Jev returns a probability with every answer. Its stated probabilities were ⟨over/under⟩-confident: mean stated probability ⟨…⟩ against accuracy ⟨…⟩; ECE ⟨…⟩ (95% CI ⟨…⟩); AUROC for separating right from wrong answers ⟨…⟩.

Deferring fields below a confidence of ⟨τ, chosen on T1⟩ kept ⟨…⟩% of fields automatic at ⟨…⟩% accuracy on T2 ∪ T3. Only ⟨…⟩% of whole cases were fully automatic, because every field in a case has to clear the threshold. ⟨Implication: per-field confidence makes a review queue possible, but whole-case automation needs ⟨…⟩.⟩

### R9. Cost, latency and deployment (Extended Data Fig. `cost`, Table `ops`)

**Sample text.** The most accurate on-premises system (⟨…⟩) ran at ⟨…⟩ s per case on ⟨hardware⟩, ⟨…⟩ GPU-hours per 1,000 cases. That was ⟨…⟩ points below / above the best API system. ⟨One sentence on PHI.⟩

### R10. Sensitivity and contamination (Extended Data Tables `sensitivity`, `subgroups`; Supplementary Table ⟨contamination⟩)

**Sample text.** The ranking of families was ⟨unchanged / changed for …⟩ under all six pre-specified sensitivity analyses (S1–S6). In the completion probe, ⟨k⟩ of ⟨n⟩ open models continued MedCalc-Bench notes verbatim more often than fresh paraphrases. ⟨Excluding them from T5 changed … (S7).⟩

---

## Discussion (outline)

1. **Principal findings.** Three sentences, one per main result: family ranking on new wording; robustness and safety; extract-then-compute against agents.
2. **Extraction is ⟨/ is not⟩ the remaining bottleneck.** Once arithmetic and criteria are fixed by the API, ⟨…⟩% of cases still get a wrong score. Clinician agreement bounds what is achievable; the ⟨…⟩-point gap to clinicians is the headroom.
3. **What memorisable benchmarks hide.** A template-inversion baseline solves template vignettes completely. Calculator benchmarks built from fixed phrases (including the parent study's) overstate extraction ability. Recommendation: report a template-inversion baseline with any templated benchmark.
4. **Schema design is a clinical decision.** Where the threshold lives (extractor or API) changes accuracy (R6). OpenMedCalc's own range checks both catch errors and reject real patients (MELD-Na sodium). Typed APIs should state units, ranges and an explicit "not documented" value.
5. **Safety.** Prompt injection inside a note is a realistic attack: notes contain patient-authored and copied text. Systems that follow instructions are exposed; typed extraction with validation limits the damage to ⟨…⟩. Deleted-finding results speak to hallucination under expectation.
6. **Selective automation.** Calibrated per-field confidence enables a review queue. Whole-case automation rates are much lower than field-level ones (R8).
7. **Deployment.** On-premises options and their accuracy cost; PHI.
8. **Comparison with prior work.** The parent study (Goodell et al., 2025); MedCalc-Bench (Khandekar et al., 2024) and its verified release; clinical information-extraction benchmarks; NuExtract and constrained decoding.
9. **Limitations.**
   - Synthetic and case-report text, not EHR notes (T6 not run).
   - The calculator set is limited to OpenMedCalc's 10.
   - Gold labels for T1–T4 come from generator metadata; independent clinician review covers only a sample.
   - Possible contamination of T5 (probe in R10).
   - API models are a snapshot.
   - Best-of-family selection uses the primary sets.
   - Clinicians self-timed.
10. **Conclusion.** Two sentences.

---

## Methods (sample text; condensed from the [analysis plan](extraction_analysis_plan.md))

### Calculators and request schemas

We used the ten OpenMedCalc calculators of the parent study: ARISCAT, Caprini, Charlson Comorbidity Index, GAD-7, HAS-BLED, MELD-Na, NIHSS, PSI/PORT, SOFA and Wells DVT. Each has a Pydantic request model with typed fields (boolean, enumerated, integer, decimal) and physiological range checks.

- **Decision schema.** This is the request model unchanged.
- **Raw-facts schema** (HAS-BLED). It replaces threshold booleans with the underlying values (systolic pressure, creatinine, weekly drinks, age), and the API applies the thresholds.
- **Scoring.** All scores were computed by the OpenMedCalc functions, loaded from source without the web application.
- **Fixes.** We corrected two calculator defects found during gold-label construction (Supplementary Note 1).

### Test sets

| Set | n | Built from | Role |
|---|---|---|---|
| T1 | 1,000 | 100 per calculator, from the re-implemented generator with seeds spaced 1,000 apart (the original seeding correlates neighbouring cases) | Comparability only |
| A409 | 397 | The parent study's released incorrect answers, matched to their seeds | Comparability only |
| T2 | ⟨≈1,000⟩ | Vignettes whose option sentences are rewritten with new phrasings, generated by a model family not under test and approved by a clinician | Primary |
| T3 | ⟨≈1,000⟩ | Whole-note paraphrases of T1 cases, approved by a clinician | Primary |
| T4 | 2,557 | The first 50 T1 cases per calculator under eight perturbations (below) | Robustness and safety |
| T5 | 637 | MedCalc-Bench Verified (pinned revision; CC-BY-SA-4.0) for the seven shared calculators, with its parameter annotations mapped to our schemas | External |

**T4 perturbations:**

- distractor sentences, including relatives' conditions (452);
- unit changes (200);
- conflicting older values (150);
- clinical abbreviations (332);
- sentence reordering (250);
- embedding in a long note (485);
- prompt injection (452);
- omission of one scored finding (236; the gold for that criterion becomes "not documented").

**T5 details.** Criteria not annotated by MedCalc-Bench were labelled "not documented", which is 21% of field instances. **Training data.** Systems that are trained used a separate generated pool, from which every vignette that also appears in a test set was removed.

### Systems

All systems produce the same request, which is validated before scoring.

- **(A) Rules.** Regular expressions with negation and attribution handling.
- **(B) Fine-tuned encoders.** Bio_ClinicalBERT, BiomedBERT, GatorTron-base and ModernBERT-base, each with one classification head per categorical field and an extractive span head for numbers.
- **(C) NuExtract.** NuExtract 2.0 (2B, 8B) and NuExtract3, run zero-shot, 3-shot, and with JSON-schema constrained decoding.
- **(D) Instruction-tuned LLMs.** Llama 3.1 8B and 70B, Qwen2.5 7B and 72B, Qwen3 8B and 32B, and GPT-4o ⟨snapshot⟩. They use JSON-schema constrained decoding, temperature 0, and the same prompt with field descriptions taken verbatim from the schema.
- **(F) Jev.** A typed decision model (TypeSafe AI, versions 1.14.0 and 1.13.0). Each field is asked as a typed question, and numbers are chosen from regex-found candidates (Supplementary Table ⟨mapping⟩).
- **Baselines:** an oracle (harness check), majority class per field, and template inversion.
- **Clinician arm.** Two clinicians, blind to gold and to each other, completed the same request for a stratified sample of 10 cases per calculator per set (T2, T3, T5) using structured sheets, and recorded their time.
- **Compute.** Open models ran on the Killarney cluster (PAICE; NVIDIA L40S) with vLLM, pinned to Hugging Face commits.

### Outcomes

- **Primary endpoints, on T2 ∪ T3:**
  - final-score accuracy (OpenMedCalc on the extracted request equals the gold score);
  - macro field accuracy (fields averaged within calculator, then across calculators). Decimals count as correct within 1%.
- **Secondary endpoints:**
  - all-fields-correct rate;
  - risk-band agreement (provisional bands, Supplementary Table ⟨bands⟩);
  - absolute score error;
  - schema validity;
  - accuracy by field type and documentation status;
  - latency and cost.
- **Invalid outputs** counted as wrong on every field.

### Statistical analysis

- **Clusters.** Rows were clustered by base case: a T1 case, its T3 paraphrase and its T4 perturbations form one cluster, and MedCalc-Bench rows sharing a note form one cluster.
- **Confidence intervals.** 95% percentile intervals from 2,000 cluster-bootstrap resamples.
- **Paired comparisons.** Differences between systems on identical cases are reported with paired cluster-bootstrap CIs and tested with the exact McNemar test (final-score accuracy).
- **Multiplicity.** Holm correction within four pre-specified families:
  1. the best system per family against each other;
  2. raw-facts against decision schema within each system;
  3. extract-then-compute against the parent study's OMC arm;
  4. each family against each clinician.
- **Field-level model.** Field correctness was modelled with a population-averaged logistic regression: generalised estimating equations with robust standard errors clustered as above. The terms were system × test set, field type and calculator. The joint Wald test of the interaction tested H1–H2.
- **Calibration.** For systems reporting a probability per answer, we computed the Brier score and expected calibration error (10 bins) with bootstrap CIs, AUROC, and the area under the risk-coverage curve. Confidence thresholds were chosen on T1 and applied unchanged to the other sets.
- **Sensitivity analyses (pre-specified):** S1 invalid outputs excluded; S2 calculators with open scoring decisions removed; S3 exact decimal matching; S4–S5 T5 with and without OpenMedCalc–MedCalc-Bench disagreements; S6 exchangeable working correlation; S7 contamination-flagged models excluded.
- **Power.** With 2,000 primary cases, a paired difference of 2.0 points is detectable with 80% power at 10% discordance (2.7 points at the strictest Holm step).
- **Pre-registration.** The analysis was frozen before real predictions were graded ⟨registration link⟩.

### Contamination probe

Each open model received the first half of each note as raw text and decoded 64 tokens greedily, for 200 T5 notes and 200 T3 paraphrases. We compared the output with the true continuation (ROUGE-L, longest shared word sequence; verbatim if ≥13 words), following Golchin & Surdeanu (2024).

### Error annotation

Two annotators coded a stratified sample of 300 field errors, blind to system: missed, hallucinated, negation, time or unit, threshold, ordinal or laterality, schema, and suspected gold error. We report Cohen's κ, and disagreements were adjudicated.

---

## Data availability (draft)

- **Committed in the repository:** T1, A409 and T4 with gold labels, the build reports and SHA-256 manifest, all predictions and evaluation outputs ⟨archive DOI⟩, and the source data for every figure.
- **T5** is derived from MedCalc-Bench Verified (CC-BY-SA-4.0) and is not redistributed. It is rebuilt from the pinned revision with one command and verified against the manifest.
- **T2/T3** are released with the clinician-approved review sheets.

## Code availability (draft)

All code is at ⟨repository URL, release tag, Zenodo DOI⟩: generator, gold mappers, extractors, evaluation, statistics, figures, and the Slurm jobs. `REPRODUCE.md` lists every command, and model revisions are pinned in `models.lock.tsv`.

---

## Display items

**Main:**

| # | Content | Key in `figures.py` |
|---|---|---|
| Fig. 1 | Study design and flow of cases | `flow` (+ design schematic) |
| Fig. 2 | Final-score accuracy by system and test set, with clinician reference | `headline` |
| Fig. 3 | Paired differences between families | `forest` |
| Fig. 4 | Error taxonomy by family | `error_taxonomy` |
| Fig. 5 | Robustness and safety perturbations | `t4_robustness` |
| Fig. 6 | Extract-then-compute vs parent-study arms | `paper_arms` |
| Fig. 7 | Confidence: risk-coverage and reliability | `confidence` |
| Table 1 | Test sets | `datasets` |
| Table 2 | Primary endpoints with CIs | `primary` |

**Extended Data:** `generalisation`, `field_heatmap`, `field_kind`, `assertion`, `schema_variant`, `cost`, `cpu_baselines`, `a409_attribution`; and the tables `power`, `pairwise`, `clinician`, `per_calculator`, `band`, `subgroups`, `sensitivity`, `variant_paired`, `model`, `model_terms`, `calibration`, `ops`.

**Supplementary:** Note 1 (data-quality findings), Jev question mapping, prompts and schemas, risk bands, contamination probe, TRIPOD-LLM checklist.

---

## References (to verify and complete; citation style to the journal's)

1. Goodell AJ, et al. Large language model agents can use tools to perform clinical calculation. *npj Digit Med.* 2025. doi:10.1038/s41746-025-01475-8.
2. Khandekar N, et al. MedCalc-Bench: evaluating large language models for medical calculations. *NeurIPS Datasets and Benchmarks.* 2024. ⟨Also cite the Verified release as dataset: nsk7153/MedCalc-Bench-Verified, revision 58fc3de.⟩
3. Gallifant J, et al. The TRIPOD-LLM reporting guideline for studies using large language models. *Nat Med.* 2025.
4. Liang K-Y, Zeger SL. Longitudinal data analysis using generalized linear models. *Biometrika.* 1986;73:13–22.
5. Davison AC, Hinkley DV. *Bootstrap Methods and their Application.* Cambridge University Press; 1997. (Cluster resampling.)
6. McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika.* 1947;12:153–157.
7. Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat.* 1979;6:65–70.
8. Connor RJ. Sample size for testing differences in proportions for the paired-sample design. *Biometrics.* 1987;43:207–211.
9. Guo C, et al. On calibration of modern neural networks. *ICML.* 2017.
10. Geifman Y, El-Yaniv R. Selective classification for deep neural networks. *NeurIPS.* 2017.
11. Golchin S, Surdeanu M. Time travel in LLMs: tracing data contamination in large language models. *ICLR.* 2024.
12. Alsentzer E, et al. Publicly available clinical BERT embeddings. *Clinical NLP Workshop.* 2019.
13. Gu Y, et al. Domain-specific language model pretraining for biomedical natural language processing. *ACM Trans Comput Healthc.* 2021.
14. Yang X, et al. A large language model for electronic health records (GatorTron). *npj Digit Med.* 2022.
15. Warner B, et al. Smarter, better, faster, longer: a modern bidirectional encoder (ModernBERT). 2024. arXiv.
16. Kwon W, et al. Efficient memory management for large language model serving with PagedAttention (vLLM). *SOSP.* 2023.
17. NuMind. NuExtract 2.0 / NuExtract3 model cards. ⟨URL, access date⟩.
18. TypeSafe AI. Jev / System One API documentation. ⟨URL, access date⟩.
19. ⟨Clinical source papers for each calculator (HAS-BLED, Wells, SOFA, MELD-Na, CCI, Caprini, ARISCAT, PSI/PORT, GAD-7, NIHSS).⟩
