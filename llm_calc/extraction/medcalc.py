"""
T5: external validity on MedCalc-Bench Verified (Khandekar et al.; Verified release
by the same group's follow-up), restricted to the calculators shared with this
benchmark: Caprini, CCI, HAS-BLED, MELD-Na, PSI/PORT, SOFA and Wells DVT.

Each MedCalc-Bench row has a patient note (PubMed case-report excerpts or
synthetic notes) and ``Relevant Entities``: the calculator inputs an annotator
extracted, with units. Criteria the note does not mention are *absent* from
that dict, so T5 is the first set here with genuinely undocumented fields.

The mappers below turn those entities into OpenMedCalc requests (variant D):
units are converted to the schema's units, and criteria that MedCalc-Bench
splits but OpenMedCalc merges are OR-ed (e.g. Wells "bedridden" / "major
surgery"). OpenMedCalc's score on the mapped request is compared with
MedCalc-Bench's own ground truth; disagreements come from calculator
definitions or annotation, and are reported, not hidden.

Licence: MedCalc-Bench is CC-BY-SA-4.0, so T5 is built on demand from a pinned
revision and is not committed to this (MIT) repository.

    python -m llm_calc.extraction build-t5            # downloads the pinned revision
    python -m llm_calc.extraction build-t5 --source DIR   # or use local CSVs
"""

import ast
import csv
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from llm_calc.extraction import gold as gold_mod
from llm_calc.extraction import omc

REPO = "nsk7153/MedCalc-Bench-Verified"
REVISION = "58fc3de568bfc53a68f32327e58f60dc6eaa17f8"
FILES = {"test": "test_data.csv", "train": "train_data.csv"}

CALCULATOR_IDS = {16: "wellsdvt", 23: "meldna", 25: "hasbled", 29: "psiport", 32: "cci", 36: "caprini", 43: "sofa"}

ABSENT = "not documented"


def download(dest: Path, revision: str = REVISION) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = dest / name
        if not target.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}"
            urllib.request.urlretrieve(url, target)
    return dest


class _Entities:
    """Accessor that records which entity set each field (or that it was absent)."""

    def __init__(self, entities: Dict):
        self.e = entities
        self.sources: Dict[str, str] = {}

    def has(self, key: str) -> bool:
        return key in self.e

    def flag(self, field: str, *keys: str) -> bool:
        present = [k for k in keys if k in self.e]
        self.sources[field] = " | ".join(present) if present else ABSENT
        return any(bool(self.e[k]) for k in present)

    def value(self, field: str, key: str, convert: Callable = None, default=None):
        if key not in self.e:
            self.sources[field] = ABSENT
            return default
        self.sources[field] = key
        raw = self.e[key]
        number, unit = (raw[0], raw[1]) if isinstance(raw, list) else (raw, None)
        return convert(number, unit) if convert else number


def _same(v, unit):
    return float(v)


def _creatinine(v, unit):
    return float(v) / 88.4 if unit and "mol" in unit else float(v)


def _temperature(v, unit):
    return (float(v) - 32) * 5 / 9 if unit and "fahrenheit" in unit else float(v)


# ---------------------------------------------------------------- mappers


def wellsdvt(x: _Entities) -> Dict:
    return {
        "active_cancer": x.flag("active_cancer", "Active cancer"),
        "bedridden_or_major_surgery_recently": x.flag(
            "bedridden_or_major_surgery_recently", "Bedridden recently >3 days", "Major surgery within 12 weeks"),
        "calf_swelling": x.flag("calf_swelling", "Calf swelling >3 centimeters compared to the other leg"),
        "collateral_veins": x.flag("collateral_veins", "Collateral (nonvaricose) superficial veins present"),
        "entire_leg_swollen": x.flag("entire_leg_swollen", "Entire Leg Swollen"),
        "localized_tenderness": x.flag("localized_tenderness", "Localized tenderness along the deep venous system"),
        "pitting_edema": x.flag("pitting_edema", "Pitting edema, confined to symptomatic leg"),
        "paralysis_paresis_or_plaster": x.flag(
            "paralysis_paresis_or_plaster", "Paralysis, paresis, or recent plaster immobilization of the lower extremity"),
        "previous_dvt": x.flag("previous_dvt", "Previously documented Deep Vein Thrombosis"),
        "alternative_diagnosis_as_likely": x.flag(
            "alternative_diagnosis_as_likely", "Alternative diagnosis to Deep Vein Thrombosis as likely or more likely"),
    }


def meldna(x: _Entities) -> Dict:
    return {
        "is_on_dialysis": x.flag("is_on_dialysis", "Dialysis at least twice in the past week",
                                 "Continuous veno-venous hemodialysis for ≥24 hours in the past week"),
        "creatinine": x.value("creatinine", "creatinine", _creatinine),
        "bilirubin": x.value("bilirubin", "Bilirubin", _same),
        "inr": x.value("inr", "international normalized ratio", _same),
        "sodium": x.value("sodium", "Sodium", _same),
    }


def hasbled(x: _Entities) -> Dict:
    drinks = x.value("alcohol", "Number of Alcoholic Drinks Per Week", _same, default=0.0)
    return {
        "hypertension": x.flag("hypertension", "Hypertension"),
        "severe_renal_disease": x.flag("severe_renal_disease", "Renal disease criteria for the HAS-BLED rule"),
        "severe_liver_disease": x.flag("severe_liver_disease", "Liver disease criteria for the HAS-BLED rule"),
        "stroke_history": x.flag("stroke_history", "Stroke"),
        "prior_major_bleeding": x.flag("prior_major_bleeding", "Prior major bleeding or predisposition to bleeding"),
        "labile_inr": x.flag("labile_inr", "Labile international normalized ratio"),
        "age": int(x.value("age", "age", _same, default=0)),
        "drugs": x.flag("drugs", "Medication usage predisposing to bleeding"),
        "alcohol": drinks >= 8,
    }


PSI_FLAGS = {
    "nursing_home_resident": "Nursing home resident",
    "neoplastic_disease": "Neoplastic disease",
    "liver_disease": "Liver disease history",
    "congestive_heart_failure": "Congestive Heart Failure",
    "cerebrovascular_disease": "Cerebrovascular disease history",
    "renal_disease": "Renal disease history",
    "altered_mental_status": "Altered mental status",
}
PSI_VALUES = {
    "respiratory_rate": ("respiratory rate", int, _same),
    "systolic_bp": ("Systolic Blood Pressure", int, _same),
    "pulse": ("Heart Rate or Pulse", int, _same),
    "temperature": ("Temperature", float, _temperature),
    "ph": ("pH", float, _same),
    "bun": ("Blood Urea Nitrogen (BUN)", int, _same),
    "sodium_mmol_L": ("Sodium", int, _same),
    "glucose": ("Glucose", int, _same),
    "hematocrit": ("Hematocrit", int, _same),
    "pao2": ("Partial pressure of oxygen", int, _same),
}


def psiport(x: _Entities) -> Dict:
    req = {"age": int(x.value("age", "age", _same, default=0))}
    sex = x.value("sex", "sex", default="Male")
    req["sex"] = "F" if str(sex).lower().startswith("f") else "M"
    for field, key in PSI_FLAGS.items():
        req[field] = x.flag(field, key)
    for field, (key, cast, convert) in PSI_VALUES.items():
        v = x.value(field, key, convert)
        req[field] = None if v is None else cast(round(v)) if cast is int else cast(v)
    req["pleural_effusion"] = int(x.flag("pleural_effusion", "Pleural effusion on x-ray"))
    return req


def cci(x: _Entities) -> Dict:
    liver = {"none": "NO", "mild": "YES_MILD", "moderate to severe": "YES_MODERATE_SEVERE"}
    diabetes = {"none or diet-controlled": "NONE", "uncomplicated": "UNCOMPLICATED", "end-organ damage": "END_ORGAN_DAMAGE"}
    tumor = {"none": "NO", "localized": "YES_LOCALIZED", "metastatic": "YES_WITH_METASTASIS"}
    return {
        "age": int(x.value("age", "age", _same, default=0)),
        "myocardial_infarction": x.flag("myocardial_infarction", "Myocardial infarction"),
        "congestive_heart_failure": x.flag("congestive_heart_failure", "Congestive Heart Failure"),
        "peripheral_vascular_disease": x.flag("peripheral_vascular_disease", "Peripheral vascular disease"),
        "cerebrovascular_disease": x.flag("cerebrovascular_disease", "Cerebrovascular Accident",
                                          "Transient Ischemic Attacks History"),
        "dementia": x.flag("dementia", "Dementia"),
        "chronic_pulmonary_disease": x.flag("chronic_pulmonary_disease", "Chronic Pulmonary Disease"),
        "rheumatic_disease": x.flag("rheumatic_disease", "Connective tissue disease"),
        "peptic_ulcer_disease": x.flag("peptic_ulcer_disease", "Peptic ulcer disease"),
        "liver_disease": liver[x.value("liver_disease", "Liver disease severity", default="none")],
        "diabetes": diabetes[x.value("diabetes", "Diabetes mellitus", default="none or diet-controlled")],
        "hemiplegia_or_paraplegia": x.flag("hemiplegia_or_paraplegia", "Hemiplegia"),
        "renal_disease": x.flag("renal_disease", "Moderate to severe Chronic Kidney Disease"),
        "solid_tumor": tumor[x.value("solid_tumor", "Solid tumor", default="none")],
        "leukemia": x.flag("leukemia", "Leukemia"),
        "lymphoma": x.flag("lymphoma", "Lymphoma"),
        "aids_hiv": x.flag("aids_hiv", "AIDS"),
    }


# Caprini 2005 lists laparoscopic (>45 min) and arthroscopic surgery at 2 points,
# which OpenMedCalc folds into "major surgery (not lower extremity)"
CAPRINI_SURGERY = {
    "minor": "minor surgery",
    "major": "major surgery (not lower extremity)",
    "laparoscopic": "major surgery (not lower extremity)",
    "arthroscopic": "major surgery (not lower extremity)",
    "elective major lower extremity arthroplasty": "major lower extremity surgery",
}
CAPRINI_MOBILITY = {
    "normal": "normal/ambulatory",
    "on bed rest": "bedrest < 72 hours and/or only walking in room",
    "confined to bed >72 hours": "confined to bed >72 hours",
}
CAPRINI_FLAGS = {
    "recent_major_surgery": ("Major Surgery in the last month",),
    "recent_chf": ("Congestive Heart Failure in the last month",),
    "recent_sepsis": ("Sepsis in the last month",),
    "recent_pna": ("Pneumonia in the last month",),
    "recent_preg": (),
    "recent_cast": ("Immobilizing plaster cast in the last month",),
    "recent_lower_extremity_or_hip_or_pelvis_fracture": ("Hip, pelvis, or leg fracture in the last month",),
    "recent_stroke": ("Stroke in the last month",),
    "recent_trauma": ("Multiple trauma in the last month",),
    "recent_spinal_cord_injury": ("Acute spinal cord injury causing paralysis in the last month",),
    "varicose_veins": ("Varicose veins",),
    "current_swollen_legs": ("Current swollen legs",),
    "current_central_venous_access": ("Current central venous access",),
    "history_of_dvt_pe": ("Previously documented Deep Vein Thrombosis", "Previously Documented Pulmonary Embolism"),
    "family_history_of_thrombosis": ("Family history of thrombosis",),
    "positive_factor_v_leiden": ("Positive Factor V Leiden",),
    "positive_prothrombin_20210a": ("Positive prothrombin 20210A",),
    "elevated_serum_homocysteine": ("Elevated serum homocysteine",),
    "positive_lupus_anticoagulant": ("Positive lupus anticoagulant",),
    "elevated_anticardiolipin_antibody": ("Elevated anticardiolipin antibody",),
    "heparin_induced_thrombocytopenia": ("Heparin-induced thrombocytopenia",),
    "other_congenital_or_acquired_thrombophilia": ("Other congenital or acquired thrombophilia",),
    "history_of_inflammatory_bowel_disease": ("History of inflammatory bowel disease",),
    "acute_mi": ("Acute Myocardial infarction",),
    "copd": ("Chronic Obstructive Pulmonary Disease",),
    "present_or_previous_malignancy": ("Present or previous malignancy",),
    "other_risk_factors": (),
}


def caprini(x: _Entities) -> Dict:
    req = {"age": int(x.value("age", "age", _same, default=0))}
    sex = x.value("sex", "sex", default="Male")
    req["sex"] = "F" if str(sex).lower().startswith("f") else "M"
    surgery = x.value("type_of_surgery", "Surgery Type", default=None)
    req["type_of_surgery"] = CAPRINI_SURGERY[surgery] if surgery else "not scheduled for surgery"
    req["mobility"] = CAPRINI_MOBILITY[x.value("mobility", "Mobility", default="normal")]
    req["bmi"] = int(round(x.value("bmi", "Body Mass Index (BMI)", _same, default=0)))
    for field, keys in CAPRINI_FLAGS.items():
        req[field] = x.flag(field, *keys)
    return req


def sofa(x: _Entities) -> Dict:
    sbp = x.value("map", "Systolic Blood Pressure", _same)
    dbp = x.e.get("Diastolic Blood Pressure")
    if sbp is not None and dbp is not None:
        mean_ap = (sbp + 2 * float(dbp[0] if isinstance(dbp, list) else dbp)) / 3
        x.sources["map"] = "Systolic Blood Pressure + Diastolic Blood Pressure"
    else:
        hypotension = x.flag("map", "Hypotension")
        mean_ap = 65.0 if hypotension else 80.0
    platelets = x.value("platelets", "Platelet count", _same)
    return {
        "pao2": x.value("pao2", "PaO2", _same),
        "fio2": x.value("fio2", "FiO2", _same),
        "mechanical_ventilation": x.flag("mechanical_ventilation", "On mechanical ventilation",
                                         "Continuous positive airway pressure"),
        # MedCalc-Bench gives platelets per µL; the schema wants ×10³/µL
        "platelets": None if platelets is None else int(round(platelets / 1000)),
        "glasgow_coma_scale": None if x.value("glasgow_coma_scale", "Glasgow Coma Score") is None
        else int(x.value("glasgow_coma_scale", "Glasgow Coma Score")),
        "bilirubin": x.value("bilirubin", "Bilirubin", _same),
        "map": round(mean_ap, 1),
        "dopamine_dose": x.value("dopamine_dose", "DOPamine", _same, default=0.0),
        "dobutamine_dose": x.value("dobutamine_dose", "DOBUTamine", _same, default=0.0),
        "epinephrine_dose": x.value("epinephrine_dose", "EPINEPHrine", _same, default=0.0),
        "norepinephrine_dose": x.value("norepinephrine_dose", "norEPINEPHrine", _same, default=0.0),
        "creatinine": x.value("creatinine", "creatinine", _creatinine),
        "urine_output": x.value("urine_output", "Urine Output", _same),
    }


MAPPERS: Dict[str, Callable[[_Entities], Dict]] = {
    "wellsdvt": wellsdvt, "meldna": meldna, "hasbled": hasbled, "psiport": psiport,
    "cci": cci, "caprini": caprini, "sofa": sofa,
}


def _answer(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def build_t5(source: Path, splits=("test",)) -> Tuple[List[Dict], Dict]:
    records, report = [], {"rows": 0, "invalid_request": 0, "score_agrees": 0, "score_disagrees": 0,
                           "by_calculator": {}}
    per_calc = {}
    for split in splits:
        with open(Path(source) / FILES[split], newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["Calculator ID"]) in CALCULATOR_IDS]
        for r in rows:
            calc = CALCULATOR_IDS[int(r["Calculator ID"])]
            report["rows"] += 1
            c = per_calc.setdefault(calc, Counter())
            x = _Entities(ast.literal_eval(r["Relevant Entities"]))
            request = MAPPERS[calc](x)
            meta = {"source": f"{REPO}@{REVISION[:12]}", "split": split, "row": r["Row Number"],
                    "note_id": r["Note ID"], "note_type": r["Note Type"],
                    "medcalc_answer": r["Ground Truth Answer"]}
            try:
                validated = omc.request_model(calc)(**request).model_dump(mode="json")
                score = gold_mod.score(calc, validated)
            except Exception as e:  # noqa: BLE001 - recorded and excluded
                report["invalid_request"] += 1
                c["invalid_request"] += 1
                meta["invalid_request"] = str(e).splitlines()[0]
                continue
            answer = _answer(r["Ground Truth Answer"])
            agrees = answer is not None and abs(score - answer) < 0.5
            report["score_agrees" if agrees else "score_disagrees"] += 1
            c["agrees" if agrees else "disagrees"] += 1
            meta["score_agrees"] = agrees
            records.append({
                "id": f"T5-{calc}-{split}-{r['Row Number']}",
                "test_set": "T5",
                "calculator": calc,
                "seed": None,
                "template_index": None,
                "vignette": r["Patient Note"],
                "gold": {"D": validated},
                "field_sources": {"D": {f: x.sources.get(f, ABSENT) for f in validated}},
                "gold_score": score,
                "original_score": answer,
                "options": [],
                "meta": meta,
            })
    report["by_calculator"] = {k: dict(v) for k, v in per_calc.items()}
    return records, report
