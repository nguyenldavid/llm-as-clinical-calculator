"""
Grade extractor predictions against gold requests.

A prediction file is JSONL with one object per case:

    {"id", "system", "variant", "output": {...} | null, "raw": str, "error": str | null,
     "latency_s": float}

``output`` is validated against the calculator's request model. A prediction
that fails validation (including the request models' physiological range
checks) counts as wrong on every field and is reported as a schema-validity
failure. Valid predictions are scored with OpenMedCalc.

A field is correct when it equals the gold value; float fields may differ by up
to ``FLOAT_REL_TOL`` (unit conversions round). ``correct_exact`` is also kept.
"""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

from llm_calc.extraction import omc
from llm_calc.extraction.schemas import field_specs, model_for
from llm_calc.extraction.schemas_raw import RAW_MODELS

FLOAT_REL_TOL = 0.01

# Provisional risk bands, taken from the category logic inside each OpenMedCalc
# calculator (lower bound of each band). Pre-specifying clinical cut-offs is an
# open decision in the analysis plan; change these once a clinician signs off.
RISK_BANDS: Dict[str, List[float]] = {
    "caprini": [1, 3, 5, 7, 9],
    "wellsdvt": [1, 3],
    "meldna": [10, 20, 30, 40],
    "sofa": [2, 4, 6, 8, 10, 12],
    "cci": [1, 3, 5],
    "ariscat": [26, 45],
    "psiport": [71, 91, 131],  # classes I and II are merged (class I also depends on age)
    "gad7": [5, 10, 15],
    "hasbled": [2, 3, 4],
    "nihss": [1, 5, 16, 21],
}


def risk_band(calculator: str, score: Optional[float]) -> Optional[int]:
    if score is None:
        return None
    return sum(score >= lb for lb in RISK_BANDS[calculator])


def _equal(kind: str, gold, pred, tolerant: bool) -> bool:
    if gold is None or pred is None:
        return gold is None and pred is None
    if kind == "float":
        if tolerant:
            return math.isclose(float(gold), float(pred), rel_tol=FLOAT_REL_TOL, abs_tol=1e-9)
        return abs(float(gold) - float(pred)) < 1e-9
    if kind == "int":
        return int(gold) == int(pred)
    return gold == pred


def validate(calculator: str, variant: str, output) -> Tuple[Optional[Dict], Optional[str]]:
    if not isinstance(output, dict):
        return None, "no JSON object"
    try:
        model = model_for(calculator, variant)(**output)
    except (ValidationError, TypeError) as e:
        return None, f"schema: {str(e).splitlines()[0]}"
    return model.model_dump(mode="json"), None


def predicted_score(calculator: str, variant: str, request: Dict) -> Optional[int]:
    try:
        if variant == "R":
            request = RAW_MODELS[calculator](**request).to_decision()
        return omc.calculate(calculator, omc.request_model(calculator)(**request))
    except Exception:  # noqa: BLE001 - a calculator crash is a wrong score, not a harness failure
        return None


def grade(records: Iterable[Dict], predictions: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Field-level and case-level result rows."""
    by_id = {p["id"]: p for p in predictions}
    field_rows, case_rows = [], []
    for rec in records:
        calc = rec["calculator"]
        pred = by_id.get(rec["id"])
        if pred is None:
            continue
        variant = pred.get("variant", "D")
        gold = rec["gold"][variant]
        output, invalid = validate(calc, variant, pred.get("output"))
        specs = field_specs(calc, variant)
        n_correct = 0
        for f in specs:
            g, p = gold.get(f.name), (output or {}).get(f.name)
            exact = output is not None and _equal(f.kind, g, p, tolerant=False)
            # primary: floats within FLOAT_REL_TOL (unit conversions round), everything else exact
            correct = output is not None and _equal(f.kind, g, p, tolerant=True)
            n_correct += correct
            field_rows.append({
                "id": rec["id"], "test_set": rec["test_set"], "calculator": calc,
                "system": pred["system"], "variant": variant, "field": f.name, "kind": f.kind,
                "gold": g, "pred": p, "correct": correct, "correct_exact": exact,
                "source": rec["field_sources"][variant].get(f.name),
            })
        score = predicted_score(calc, variant, output) if output is not None else None
        gold_score = rec["gold_score"]
        case_rows.append({
            "id": rec["id"], "test_set": rec["test_set"], "calculator": calc,
            "system": pred["system"], "variant": variant,
            "valid": output is not None, "invalid_reason": invalid or pred.get("error"),
            "n_fields": len(specs), "n_correct": n_correct, "all_fields_correct": n_correct == len(specs),
            "gold_score": gold_score, "pred_score": score,
            "score_correct": score is not None and score == gold_score,
            "abs_error": None if score is None else abs(score - gold_score),
            "band_correct": score is not None and risk_band(calc, score) == risk_band(calc, gold_score),
            "latency_s": pred.get("latency_s"),
        })
    return field_rows, case_rows


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def summarise(field_rows: List[Dict], case_rows: List[Dict], by=("system", "variant", "test_set")) -> List[Dict]:
    """One row per group, plus per-calculator rows (``calculator`` set)."""
    cases = defaultdict(list)
    fields = defaultdict(list)
    for r in case_rows:
        key = tuple(r[k] for k in by)
        cases[key].append(r)
        cases[key + (r["calculator"],)].append(r)
    for r in field_rows:
        key = tuple(r[k] for k in by)
        fields[key].append(r)
        fields[key + (r["calculator"],)].append(r)

    out = []
    for key, rows in sorted(cases.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        frows = fields[key]
        per_calc_field = defaultdict(lambda: defaultdict(list))
        for f in frows:
            per_calc_field[f["calculator"]][f["field"]].append(f["correct"])
        # macro: mean over calculators of the mean over fields of field accuracy
        macro = _mean(_mean(_mean(v) for v in flds.values()) for flds in per_calc_field.values())
        row = dict(zip(by, key[: len(by)]))
        row["calculator"] = key[len(by)] if len(key) > len(by) else "ALL"
        row.update({
            "n_cases": len(rows),
            "valid_rate": _mean(r["valid"] for r in rows),
            "macro_field_acc": macro,
            "micro_field_acc": _mean(f["correct"] for f in frows),
            "micro_field_acc_exact": _mean(f["correct_exact"] for f in frows),
            "all_fields_correct": _mean(r["all_fields_correct"] for r in rows),
            "score_acc": _mean(r["score_correct"] for r in rows),
            "band_acc": _mean(r["band_correct"] for r in rows),
            "score_mae": _mean(r["abs_error"] for r in rows),
            "median_latency_s": _median(r["latency_s"] for r in rows),
        })
        out.append(row)
    return out


def field_accuracy_by_kind(field_rows: List[Dict], by=("system", "variant", "test_set")) -> List[Dict]:
    groups = defaultdict(list)
    for r in field_rows:
        groups[tuple(r[k] for k in by) + (r["kind"],)].append(r["correct"])
    return [dict(zip(by + ("kind",), k), n=len(v), acc=_mean(v)) for k, v in sorted(groups.items())]


def field_accuracy_by_documentation(field_rows: List[Dict], by=("system", "variant", "test_set")) -> List[Dict]:
    """Field accuracy split by whether the gold value is stated in the text.

    Only external sets mark undocumented fields (``source == "not documented"``,
    T5); there the gold value is the calculator's default (absent / 0).
    """
    groups = defaultdict(list)
    for r in field_rows:
        documented = r.get("source") != "not documented"
        groups[tuple(r[k] for k in by) + (documented,)].append(r["correct"])
    return [dict(zip(by + ("documented",), k), n=len(v), acc=_mean(v)) for k, v in sorted(groups.items())]
