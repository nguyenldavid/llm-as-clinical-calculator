"""
Mock results for every planned experiment, so the figures and tables can be
designed and agreed before any GPU time is spent.

Nothing here is a result. Mock *predictions* are made by corrupting the gold
requests of the real test records at an assumed per-field accuracy, then graded
by the real evaluator, so every downstream number (score, risk band, field
accuracy, CIs) is internally consistent and the figure code is exercised exactly
as it will be on real output. The assumed accuracies are arbitrary placeholders
(see ``SYSTEMS``), chosen only to make the shells legible.

T2 and T3 do not exist yet; their mock stand-ins are copies of T1 records.

    python -m llm_calc.extraction mock-results --out <dir>
    python -m llm_calc.extraction figures --eval-dir <dir> --out docs/mock_results --mock
"""

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import evaluate as ev
from llm_calc.extraction import schemas


@dataclass
class MockSystem:
    name: str
    family: str
    skill: float  # baseline log-odds of a correct field
    memorisation: float  # extra log-odds on generator-phrased text (T1, A409, T4)
    invalid_rate: float  # share of outputs with no usable JSON
    latency_s: float  # per case, on its own hardware
    deployment: str  # on-prem CPU | on-prem GPU | external API


# Placeholder parameters. They encode the plan's hypotheses (H1-H4) only so the
# shells show the intended contrasts; they are not predictions.
SYSTEMS: List[MockSystem] = [
    MockSystem("rules", "Rules", 2.2, 3.0, 0.01, 0.002, "on-prem CPU"),
    MockSystem("encoder-Bio_ClinicalBERT", "Encoder", 2.3, 3.2, 0.005, 0.01, "on-prem GPU"),
    MockSystem("encoder-BiomedBERT", "Encoder", 2.4, 3.2, 0.005, 0.01, "on-prem GPU"),
    MockSystem("encoder-gatortron-base", "Encoder", 2.5, 3.2, 0.005, 0.01, "on-prem GPU"),
    MockSystem("encoder-ModernBERT-base", "Encoder", 2.5, 3.0, 0.005, 0.01, "on-prem GPU"),
    MockSystem("nuextract:NuExtract-2.0-2B", "NuExtract", 2.7, 0.5, 0.02, 0.15, "on-prem GPU"),
    MockSystem("nuextract:NuExtract-2.0-8B", "NuExtract", 3.1, 0.5, 0.015, 0.3, "on-prem GPU"),
    MockSystem("nuextract:NuExtract-2.0-8B:3shot", "NuExtract", 3.4, 0.6, 0.01, 0.5, "on-prem GPU"),
    MockSystem("nuextract:NuExtract3", "NuExtract", 3.4, 0.5, 0.01, 0.25, "on-prem GPU"),
    MockSystem("nuextract:NuExtract3:3shot", "NuExtract", 3.7, 0.6, 0.008, 0.4, "on-prem GPU"),
    MockSystem("nuextract:NuExtract3:constrained", "NuExtract", 3.5, 0.5, 0.0, 0.3, "on-prem GPU"),
    MockSystem("vllm:Llama-3.1-8B-Instruct", "Open LLM", 2.9, 0.4, 0.0, 0.3, "on-prem GPU"),
    MockSystem("vllm:Qwen2.5-7B-Instruct", "Open LLM", 3.0, 0.4, 0.0, 0.3, "on-prem GPU"),
    MockSystem("vllm:Qwen3-8B", "Open LLM", 3.2, 0.4, 0.0, 0.3, "on-prem GPU"),
    MockSystem("vllm:Qwen3-32B", "Open LLM", 3.7, 0.4, 0.0, 0.8, "on-prem GPU"),
    MockSystem("vllm:Llama-3.1-70B-Instruct", "Open LLM", 3.8, 0.4, 0.0, 1.5, "on-prem GPU"),
    MockSystem("vllm:Llama-3.1-70B-Instruct:3shot", "Open LLM", 4.0, 0.5, 0.0, 2.0, "on-prem GPU"),
    MockSystem("vllm:Qwen2.5-72B-Instruct", "Open LLM", 3.9, 0.4, 0.0, 1.6, "on-prem GPU"),
    MockSystem("api:gpt-4o", "API LLM", 4.0, 0.4, 0.0, 2.5, "external API"),
    MockSystem("jev:jev-1.14.0", "System One", 3.6, 0.3, 0.0, 0.3, "external API"),
    MockSystem("jev:jev-1.13.0", "System One", 3.4, 0.3, 0.0, 0.3, "external API"),
]
FAMILIES = ["Rules", "Encoder", "NuExtract", "Open LLM", "API LLM", "System One"]
CONFIDENCE_FAMILIES = {"System One"}  # systems that report a confidence per answer

SET_DIFFICULTY = {"T1": 0.0, "A409": 0.3, "T4": 0.7, "T2": 1.0, "T3": 1.2, "T5": 1.7}
GENERATOR_PHRASED = {"T1", "A409", "T4"}
KIND_EFFECT = {"bool": 0.2, "enum": -0.4, "int": 0.3, "float": 0.1}
CALC_EFFECT = {"nihss": -0.4, "sofa": -0.3, "psiport": -0.1, "caprini": 0.3, "cci": 0.1}

ERROR_CODES = ["MISS", "HALLUC", "NEG", "TEMP/UNIT", "THRESH", "ORD/LAT", "SCHEMA"]
# relative weights of error codes per family (placeholders)
ERROR_WEIGHTS = {
    "Rules": [5, 1, 3, 3, 1, 2, 1],
    "Encoder": [3, 2, 3, 3, 2, 3, 1],
    "NuExtract": [3, 2, 2, 2, 3, 2, 2],
    "Open LLM": [2, 3, 2, 2, 3, 2, 0.2],
    "API LLM": [2, 3, 1, 1, 3, 2, 0.1],
    "System One": [3, 1, 3, 2, 2, 3, 0.1],
}

# the paper's end-to-end arms on T1, plus extract-then-compute (RQ2); placeholder accuracies
PAPER_ARMS = {
    "GPT-4o": {"base": 0.72, "rag": 0.77, "ci": 0.80, "rag_ci": 0.82, "omc (agent)": 0.93, "extract→compute": 0.96},
    "Llama 3.1 70B": {"base": 0.55, "rag": 0.62, "ci": 0.66, "rag_ci": 0.68, "omc (agent)": 0.86, "extract→compute": 0.94},
}


def family_of(system: str) -> str:
    for s in SYSTEMS:
        if s.name == system:
            return s.family
    return "Other"


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _corrupt(spec: schemas.FieldSpec, value, rng: random.Random):
    """A plausible wrong value for a field."""
    if spec.kind == "bool":
        return not value if value is not None else rng.random() < 0.5
    if spec.kind == "enum":
        others = [c for c in spec.choices if c != value]
        return rng.choice(others) if others else value
    if value is None:
        return 1 if spec.kind == "int" else 1.0
    factor = rng.choice([0.5, 0.8, 1.25, 2.0])
    new = value * factor if value else rng.choice([1, 2, 5])
    return int(round(new)) if spec.kind == "int" else round(float(new), 2)


def mock_predictions(records: List[Dict], system: MockSystem, variant: str, seed: int) -> List[Dict]:
    rng = random.Random(f"{system.name}:{variant}:{seed}")
    preds = []
    for r in records:
        if rng.random() < system.invalid_rate:
            preds.append({"id": r["id"], "system": system.name, "variant": variant, "output": None,
                          "raw": "", "error": "mock: unparseable", "latency_s": system.latency_s})
            continue
        gold = r["gold"][variant]
        out, confidence = {}, {}
        for spec in schemas.field_specs(r["calculator"], variant):
            logit = system.skill - SET_DIFFICULTY.get(r["test_set"], 1.0)
            if r["test_set"] in GENERATOR_PHRASED:
                logit += system.memorisation
            logit += KIND_EFFECT[spec.kind] + CALC_EFFECT.get(r["calculator"], 0.0)
            if variant == "R":
                logit += 0.4 if system.family in ("Rules", "Encoder") else 0.1
            if r["field_sources"][variant].get(spec.name) == "not documented":
                logit += 1.2 if system.family in ("Rules", "Encoder") else 0.6
            value = gold.get(spec.name)
            correct = rng.random() < _sigmoid(logit)
            out[spec.name] = value if correct else _corrupt(spec, value, rng)
            if system.family in CONFIDENCE_FAMILIES:
                # informative but imperfect confidence: higher when right
                confidence[spec.name] = rng.betavariate(7, 1.5) if correct else rng.betavariate(2.5, 2.5)
        pred = {"id": r["id"], "system": system.name, "variant": variant, "output": out, "raw": "",
                "error": None, "latency_s": system.latency_s * rng.uniform(0.8, 1.2)}
        if confidence:
            pred["confidence"] = confidence
        preds.append(pred)
    return preds


def _cap(records: List[Dict], per_calculator: int) -> List[Dict]:
    taken, out = Counter(), []
    for r in records:
        if taken[r["calculator"]] < per_calculator:
            taken[r["calculator"]] += 1
            out.append(r)
    return out


def mock_sets(data_dir: Path = ds.DATA_DIR, per_calculator: int = 40) -> List[Dict]:
    """A subsample of every set (shells need shape, not power)."""
    t1 = _cap(ds.load_set("T1", data_dir), per_calculator)
    base_ids = {r["id"] for r in _cap(t1, per_calculator // 2)}
    t4 = [r for r in ds.load_set("T4", data_dir) if r["meta"]["base_id"] in base_ids]
    records = t1 + _cap(ds.load_set("A409", data_dir), per_calculator) + t4
    for name in ("T2", "T3"):  # stand-ins until the reviewed sets exist
        records += [dict(r, id=r["id"].replace("T1-", f"{name}mock-"), test_set=name) for r in t1]
    if (data_dir / "T5.jsonl").exists():
        records += _cap(ds.load_set("T5", data_dir), per_calculator * 2)
    return records


FIELD_KEYS = ("id", "test_set", "calculator", "system", "variant", "field", "kind", "correct", "source", "confidence")


def build(out: Path, data_dir: Path = ds.DATA_DIR, seed: int = 0) -> Dict:
    records = mock_sets(data_dir)
    field_rows, case_rows = [], []
    for system in SYSTEMS:
        for variant in ("D", "R"):
            recs = [r for r in records if variant in r["gold"]]
            preds = mock_predictions(recs, system, variant, seed)
            f, c = ev.grade(recs, preds)
            field_rows += [{k: row[k] for k in FIELD_KEYS} for row in f]
            case_rows += c
    pert = {r["id"]: r["meta"].get("perturbation", "") for r in records}
    for row in field_rows + case_rows:
        row["perturbation"] = pert.get(row["id"], "")
        row["family"] = family_of(row["system"])
    out.mkdir(parents=True, exist_ok=True)
    ds.write_jsonl(case_rows, out / "cases.jsonl")
    ds.write_jsonl(field_rows, out / "fields.jsonl")

    rng = random.Random(seed)
    # error taxonomy: one code per wrong field, as an annotator would assign
    errors = Counter()
    for f in field_rows:
        if not f["correct"] and f["variant"] == "D":
            fam = family_of(f["system"])
            errors[(fam, rng.choices(ERROR_CODES, ERROR_WEIGHTS[fam])[0])] += 1
    with open(out / "error_codes.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["family", "code", "n"])
        for (fam, code), n in sorted(errors.items()):
            w.writerow([fam, code, n])

    with open(out / "paper_arms.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["model", "arm", "n", "correct"])
        for model, arms in PAPER_ARMS.items():
            for arm, acc in arms.items():
                n = 1000
                correct = sum(rng.random() < acc for _ in range(n))
                w.writerow([model, arm, n, correct])

    with open(out / "systems.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["system", "family", "deployment"])
        for s in SYSTEMS:
            w.writerow([s.name, s.family, s.deployment])
    (out / "MOCK").write_text("Every file in this directory is mock data generated by llm_calc.extraction.mock.\n")
    return {"cases": len(case_rows), "fields": len(field_rows)}
