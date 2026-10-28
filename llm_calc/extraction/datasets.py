"""
Benchmark datasets with per-field gold labels.

Every record is one JSON object per line:

    id, test_set, calculator, seed, template_index, vignette,
    gold          {"D": decision request, "R": raw-facts request (where defined)}
    field_sources {variant: {field: option name or "template"}}
    gold_score    OpenMedCalc score of gold["D"]
    original_score  ground truth as the original experiment computed it
    options       names of the sampled criterion options
    meta          set-specific extras (e.g. the Abacus-409 model answer)

Sets
----
T1      fresh cases from the original generator (comparability with the paper)
TRAIN   training pool for fine-tuned systems; never overlaps a test vignette
A409    the released Abacus-409 failures matched back to their seeds
T2      held-out option phrasings (needs reviewed phrasings, see phrasings.py)
T3      whole-vignette paraphrases (needs reviewed paraphrases)
T4      controlled perturbations of T1 (see perturb.py)

Seeds are spaced ``SEED_STRIDE`` apart: the original generator seeds criterion
``k`` of case ``s`` with ``s + k``, so consecutive seeds share random streams.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from llm_calc.extraction import generator as g
from llm_calc.extraction import gold as gold_mod
from llm_calc.extraction.schemas_raw import RAW_MAPPERS, RAW_MODELS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "llm_calc" / "data" / "extraction"
ABACUS409 = ROOT / "datasets" / "abacus409.csv"

SEED_STRIDE = 1000
T1_SEED_BASE = 20_000_000
TRAIN_SEED_BASE = 90_000_000
# the original experiment used case ids well below this
ORIGINAL_SEED_SEARCH = range(0, 2000)


def record_for(case: g.GeneratedCase, test_set: str, case_id: str, meta: Optional[Dict] = None) -> Dict:
    d = gold_mod.gold_for(case)
    gold = {"D": d.request}
    sources = {"D": d.sources}
    if case.calculator in RAW_MAPPERS:
        r = RAW_MAPPERS[case.calculator](case.options)
        gold["R"] = RAW_MODELS[case.calculator](**r.request).model_dump(mode="json")
        sources["R"] = r.sources
    return {
        "id": case_id,
        "test_set": test_set,
        "calculator": case.calculator,
        "seed": case.seed,
        "template_index": case.template_index,
        "vignette": case.vignette,
        "gold": gold,
        "field_sources": sources,
        "gold_score": gold_mod.score(case.calculator, d.request),
        "original_score": int(float(case.original_score)),
        "options": [o.name for o in case.options],
        "meta": meta or {},
    }


def _generated_set(name: str, seed_base: int, per_calculator: int, calculators: List[str]) -> List[Dict]:
    records = []
    for ci, calc in enumerate(calculators):
        for i in range(per_calculator):
            seed = seed_base + (ci * per_calculator + i) * SEED_STRIDE
            records.append(record_for(g.build_case(calc, seed), name, f"{name}-{calc}-{i:04d}"))
    return records


def build_t1(per_calculator: int = 100, calculators: List[str] = g.CALCULATOR_ORDER) -> List[Dict]:
    return _generated_set("T1", T1_SEED_BASE, per_calculator, calculators)


def build_train(
    per_calculator: int, exclude_vignettes: Iterable[str], calculators: List[str] = g.CALCULATOR_ORDER
) -> List[Dict]:
    """Training pool with any vignette that also occurs in a test set removed.

    The templated space is small (HAS-BLED has a few thousand distinct
    vignettes), so exact duplicates of test cases are common without this.
    """
    held_out = {g._normalise(v) for v in exclude_vignettes}
    pool = _generated_set("TRAIN", TRAIN_SEED_BASE, per_calculator, calculators)
    return [r for r in pool if g._normalise(r["vignette"]) not in held_out]


def build_a409(search: range = ORIGINAL_SEED_SEARCH) -> Dict:
    """Match Abacus-409 rows to generated cases; returns records plus a report."""
    index = g.seed_index(g.CALCULATOR_ORDER, search)
    records, report = [], {"rows": 0, "empty_vignette": 0, "unmatched": 0, "matched": 0,
                           "ambiguous_gold": 0, "ground_truth_disagrees": 0}
    with open(ABACUS409, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        report["rows"] += 1
        text = g._normalise(row["vignette"])
        if not text:
            report["empty_vignette"] += 1
            continue
        candidates = index.get(text)
        if not candidates:
            report["unmatched"] += 1
            continue
        golds = {json.dumps(gold_mod.gold_for(c).request, sort_keys=True) for c in candidates}
        if len(golds) > 1:
            report["ambiguous_gold"] += 1
            continue
        case = candidates[0]
        meta = {k: row[k] for k in ["id", "llm_answer", "explanation", "ground_truth", "annotation",
                                    "arm", "error_class_one", "error_class_two"]}
        meta["candidate_seeds"] = sorted(c.seed for c in candidates)
        rec = record_for(case, "A409", f"A409-{row['id']}", meta)
        # keep the released vignette text, which may use historical option wording
        rec["vignette"] = row["vignette"]
        try:
            released_truth = int(float(row["ground_truth"]))
        except ValueError:
            released_truth = None
        if released_truth != rec["gold_score"]:
            report["ground_truth_disagrees"] += 1
            rec["meta"]["ground_truth_disagrees"] = True
        records.append(rec)
        report["matched"] += 1
    return {"records": records, "report": report}


def build_t4(t1_records: List[Dict], per_calculator: int = 50, seed: int = 0) -> List[Dict]:
    """Perturbed copies of the first ``per_calculator`` T1 cases of each calculator.

    Records keep the T1 case's gold labels; ``meta`` names the perturbation and the
    T1 case it came from, so results can be compared pairwise with T1.
    """
    import random

    from llm_calc.extraction import perturb

    rng = random.Random(seed)
    safety_rng = random.Random(seed + 1)
    taken: Dict[str, int] = {}
    records = []
    for base in t1_records:
        calc = base["calculator"]
        if taken.get(calc, 0) >= per_calculator:
            continue
        taken[calc] = taken.get(calc, 0) + 1
        case = g.build_case(calc, base["seed"])
        for kind in perturb.PERTURBATIONS:
            text = perturb.perturb(case, kind, rng)
            if text is None:
                continue
            rec = dict(base, id=f"T4-{kind}-{base['id']}", test_set="T4", vignette=text,
                       meta={"perturbation": kind, "base_id": base["id"]})
            records.append(rec)
        text = perturb.inject(case, safety_rng)
        if text is not None:
            records.append(dict(base, id=f"T4-injection-{base['id']}", test_set="T4", vignette=text,
                                meta={"perturbation": "injection", "base_id": base["id"]}))
        omitted = perturb.omit(case, safety_rng)
        if omitted is not None:
            records.append(_omission_record(case, omitted, base))
    return records


def _omission_record(case: g.GeneratedCase, omitted: Dict, base: Dict) -> Dict:
    """Gold for a note with one finding deleted.

    Every field sourced from the deleted criterion becomes "not documented": the
    decision schema keeps the calculator default (absent), the raw-facts schema
    records null.
    """
    import dataclasses

    new_case = dataclasses.replace(case, options=omitted["options"], vignette=omitted["text"])
    rec = record_for(new_case, "T4", f"T4-omission-{base['id']}",
                     {"perturbation": "omission", "base_id": base["id"],
                      "omitted_criterion": omitted["omitted_criterion"], "omitted_option": omitted["omitted_option"]})
    rec["vignette"] = omitted["text"]
    rec["original_score"] = rec["gold_score"]
    default_option = next(o.name for o in omitted["options"] if o.criterion_slug == omitted["omitted_criterion"])
    for variant, sources in rec["field_sources"].items():
        for field, source in sources.items():
            if source == default_option:
                sources[field] = "not documented"
                if variant == "R":
                    rec["gold"]["R"][field] = None
    return rec


def write_jsonl(records: Iterable[Dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> Iterator[Dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_set(name: str, data_dir: Path = DATA_DIR) -> List[Dict]:
    return list(read_jsonl(data_dir / f"{name}.jsonl"))
