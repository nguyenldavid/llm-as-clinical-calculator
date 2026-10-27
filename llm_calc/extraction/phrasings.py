"""
Held-out phrasings for T2 and whole-vignette paraphrases for T3.

Sheets are tab-separated (``.tsv``): ``*.csv`` is routed through Git LFS in this
repository, and review sheets should diff like ordinary text.

Workflow
--------
1. ``export_options`` writes every paraphrasable option of the ten calculators.
2. A GPU job (``scripts/paraphrase.py``, run on Killarney) writes candidate
   rewordings into a review sheet: one row per candidate, with a deterministic
   ``split`` of train or test.
3. A clinician fills the ``approved`` column (y/n) and optionally ``note``.
4. ``build_t2`` renders fresh cases with *test* phrasings; ``train_phrasings``
   supplies *train* phrasings for fine-tuned systems. The two never overlap.
5. T3 works the same way at the level of whole vignettes (``build_t3``).

Only approved rows are ever used. Options whose text is a bare value (a lab
number, an age, "man") are not paraphrased; T3 covers their surroundings.
"""

import csv
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import generator as g

REVIEW_COLUMNS = ["key", "calculator", "criterion", "option_name", "score_effect", "original",
                  "candidate", "generator", "split", "approved", "note"]
T3_COLUMNS = ["base_id", "calculator", "original", "candidate", "generator", "approved", "note"]
T2_SEED_BASE = 30_000_000


def option_key(o: g.Option) -> str:
    raw = f"{o.criterion_slug}::{o.name}::{' '.join(o.input_string.split())}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def paraphrasable(o: g.Option) -> bool:
    text = o.input_string.strip()
    return len(re.findall(r"[A-Za-z]{3,}", text)) >= 3


def export_options(path: Path, calculators: List[str] = g.CALCULATOR_ORDER) -> int:
    """Unique paraphrasable options, with the template sentence around each slot."""
    seen, rows = set(), []
    for calc in calculators:
        template = g.templates(calc)[0]
        for crit in g._placeholders(calc):
            context = next((s for s in re.split(r"(?<=[.?])\s+", template) if f"{{{crit}}}" in s), "")
            for o in g._options_for(crit):
                key = option_key(o)
                if key in seen or not paraphrasable(o):
                    continue
                seen.add(key)
                rows.append({"key": key, "calculator": calc, "criterion": crit, "option_name": o.name,
                             "score_effect": o.score_effect, "original": " ".join(o.input_string.split()),
                             "context": context})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def assign_split(key: str, candidate: str, test_fraction: float = 0.5) -> str:
    h = int(hashlib.sha1(f"{key}::{candidate}".encode()).hexdigest(), 16)
    return "test" if (h % 1000) / 1000 < test_fraction else "train"


def _approved_rows(path: Path, columns: List[str]) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    missing = set(columns) - set(rows[0]) if rows else set()
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return [r for r in rows if r["approved"].strip().lower() in {"y", "yes", "1", "true"}]


def load_phrasings(review_sheet: Path) -> Dict[str, Dict[str, List[str]]]:
    """option key -> {"train": [...], "test": [...]} from approved review rows."""
    out: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"train": [], "test": []})
    for r in _approved_rows(review_sheet, REVIEW_COLUMNS):
        out[r["key"]][r["split"]].append(r["candidate"].strip())
    return dict(out)


def rephrase_case(case: g.GeneratedCase, phrasings: Dict, split: str, rng: random.Random) -> Optional[Dict]:
    """Render ``case`` with ``split`` phrasings; None if no option could be rephrased."""
    from llm_calc.extraction.perturb import render

    overrides, rephrased = {}, 0
    for o in case.options:
        choices = phrasings.get(option_key(o), {}).get(split, [])
        if choices:
            overrides[o.criterion_slug] = rng.choice(choices)
            rephrased += 1
    eligible = sum(paraphrasable(o) for o in case.options)
    if not rephrased:
        return None
    return {"vignette": render(case, overrides), "rephrased_slots": rephrased, "eligible_slots": eligible}


def build_t2(review_sheet: Path, per_calculator: int = 100, seed: int = 0,
             calculators: List[str] = g.CALCULATOR_ORDER) -> List[Dict]:
    phrasings = load_phrasings(review_sheet)
    rng = random.Random(seed)
    records = []
    for ci, calc in enumerate(calculators):
        made, i = 0, 0
        while made < per_calculator and i < per_calculator * 20:
            case_seed = T2_SEED_BASE + (ci * per_calculator * 20 + i) * ds.SEED_STRIDE
            i += 1
            case = g.build_case(calc, case_seed)
            out = rephrase_case(case, phrasings, "test", rng)
            if out is None:
                continue
            rec = ds.record_for(case, "T2", f"T2-{calc}-{made:04d}",
                                {"rephrased_slots": out["rephrased_slots"], "eligible_slots": out["eligible_slots"]})
            rec["vignette"] = out["vignette"]
            records.append(rec)
            made += 1
    return records


def train_with_phrasings(train_records: List[Dict], review_sheet: Path, seed: int = 0) -> List[Dict]:
    """Copies of training records re-rendered with *train* phrasings (augmentation)."""
    phrasings = load_phrasings(review_sheet)
    rng = random.Random(seed)
    out = []
    for r in train_records:
        case = g.build_case(r["calculator"], r["seed"])
        res = rephrase_case(case, phrasings, "train", rng)
        if res is not None:
            out.append(dict(r, id=r["id"] + "-p", vignette=res["vignette"]))
    return out


def build_t3(paraphrase_sheet: Path, base_records: List[Dict]) -> List[Dict]:
    """Approved whole-vignette paraphrases; gold labels come from the base case."""
    base = {r["id"]: r for r in base_records}
    records = []
    for k, row in enumerate(_approved_rows(paraphrase_sheet, T3_COLUMNS)):
        b = base[row["base_id"]]
        records.append(dict(b, id=f"T3-{k:05d}-{b['id']}", test_set="T3", vignette=row["candidate"].strip(),
                            meta={"base_id": b["id"], "generator": row["generator"]}))
    return records
