"""Regression tests for the extraction benchmark (CPU only, about a minute).

    python -m pytest tests/test_extraction.py -q
"""

import pandas as pd
import pytest

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import evaluate as ev
from llm_calc.extraction import generator as g
from llm_calc.extraction import gold, perturb, schemas
from llm_calc.extraction.extractors.baselines import Oracle, TemplateInversion
from llm_calc.extraction.extractors.rules import RulesExtractor
from llm_calc.extraction.schemas_raw import RawRequestHASBLED, hasbled_raw


@pytest.mark.parametrize("n", [1, 2, 3, 24, 46])
def test_pick_matches_pandas_sample(n):
    df = pd.DataFrame({"a": range(n)})
    for seed in range(200):
        assert df.sample(n=1, random_state=seed).a.iloc[0] == g._pick(n, seed)


@pytest.mark.parametrize("calculator", g.CALCULATOR_ORDER)
def test_gold_round_trip(calculator):
    """OpenMedCalc on the gold request reproduces the original ground truth."""
    for seed in range(200):
        ok, computed, original = gold.round_trip(g.build_case(calculator, seed))
        assert ok, (calculator, seed, computed, original)


def test_wells_negative_scores_do_not_crash():
    model = schemas.model_for("wellsdvt")
    request = {f: False for f in model.model_fields}
    request["alternative_diagnosis_as_likely"] = True
    assert gold.score("wellsdvt", request) == -2


def test_raw_hasbled_converts_to_decision_schema():
    for seed in range(200):
        case = g.build_case("hasbled", seed)
        raw = RawRequestHASBLED(**hasbled_raw(case.options).request)
        assert raw.to_decision() == gold.gold_for(case).request


def test_render_reproduces_generator_text():
    for calc in g.CALCULATOR_ORDER:
        for seed in range(50):
            case = g.build_case(calc, seed)
            assert perturb.render(case, {}) == g._normalise(case.vignette)


@pytest.fixture(scope="module")
def t1():
    return ds.load_set("T1")


def test_released_sets_are_consistent(t1):
    assert len(t1) == 1000
    assert all(r["gold_score"] == r["original_score"] for r in t1)
    a409 = ds.load_set("A409")
    assert len(a409) == 397
    assert not any(r["meta"].get("ground_truth_disagrees") for r in a409)


def _score_acc(extractor, records, variant="D"):
    preds = extractor.run(records, variant)
    _, cases = ev.grade(records, preds)
    return sum(c["score_correct"] for c in cases) / len(cases)


def test_oracle_and_template_inversion(t1):
    sample = t1[::7]
    assert _score_acc(Oracle(), sample) == 1.0
    assert _score_acc(TemplateInversion(t1), sample) == 1.0


def test_rules_run_on_every_set(t1):
    for name in ["T1", "A409", "T4"]:
        records = ds.load_set(name)[::25]
        preds = RulesExtractor().run(records, "D")
        assert all(p["output"] is not None for p in preds)


def test_sofa_ventilated_patients_score_moderate_ratios():
    """Ventilated, P/F 250: 2 respiratory points (OpenMedCalc used to give 0)."""
    base = {"pao2": 100.0, "fio2": 40.0, "mechanical_ventilation": True, "platelets": 200,
            "glasgow_coma_scale": 15, "bilirubin": 0.5, "map": 80.0, "dopamine_dose": 0.0,
            "dobutamine_dose": 0.0, "epinephrine_dose": 0.0, "norepinephrine_dose": 0.0,
            "creatinine": 0.8, "urine_output": 1500.0}
    assert gold.score("sofa", base) == 2
    assert gold.score("sofa", dict(base, mechanical_ventilation=False)) == 2
    assert gold.score("sofa", dict(base, pao2=60.0)) == 3  # P/F 150, ventilated


def test_t5_mapping_agrees_with_medcalc_bench():
    """Needs build-t5 (MedCalc-Bench is not committed); skipped otherwise."""
    path = ds.DATA_DIR / "T5.jsonl"
    if not path.exists():
        pytest.skip("T5 not built")
    t5 = list(ds.read_jsonl(path))
    agree = sum(r["meta"]["score_agrees"] for r in t5) / len(t5)
    assert agree > 0.98
