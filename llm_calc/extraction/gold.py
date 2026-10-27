"""
Gold typed requests for generated cases.

Each mapper turns the criterion options the generator sampled for a case into
the OpenMedCalc request (schema variant "D", the decision schema) and records,
for every field, which option or template sentence determined its value. The
mappings are written out explicitly so that a clinician can review them line by
line; anything unexpected raises instead of guessing.

``round_trip`` checks that OpenMedCalc applied to the gold request reproduces
the ground truth the original experiment used (sum of option points, or the
MELD-Na calculator). A mismatch means either a mapper bug or an inconsistency
in the original ground truth, and is reported rather than silently fixed.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from llm_calc.extraction import omc
from llm_calc.extraction.generator import GeneratedCase, Option

TEMPLATE = "template"  # field fixed by template wording, e.g. "no other venous or clotting disorders"


@dataclass
class Gold:
    request: Dict  # JSON-serialisable values, validated against the request model
    sources: Dict[str, str]  # field -> option name (or TEMPLATE) that set it


def _num(text: str) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"no number in {text!r}")
    return float(m.group())


def _by_criterion(options: List[Option]) -> Dict[str, Option]:
    return {o.criterion_slug: o for o in options}


class _Builder:
    def __init__(self, options: List[Option]):
        self.by = _by_criterion(options)
        self.request: Dict = {}
        self.sources: Dict[str, str] = {}

    def set(self, field: str, value, criterion: str = None):
        self.request[field] = value
        self.sources[field] = self.by[criterion].name if criterion else TEMPLATE

    def opt(self, criterion: str) -> Option:
        return self.by[criterion]

    def flag(self, field: str, criterion: str):
        """Boolean field that is true exactly when the option scores points."""
        self.set(field, self.by[criterion].score_effect != 0, criterion)

    def gold(self) -> Gold:
        return Gold(self.request, self.sources)


def _lookup(table: Dict, key, what: str):
    if key not in table:
        raise KeyError(f"unmapped {what}: {key!r}")
    return table[key]


# ---------------------------------------------------------------- Caprini

CAPRINI_SURGERY = {
    "Surgery_they_are_not_scheduled_for_surgery": "not scheduled for surgery",
    "Surgery_they_are_scheduled_for_a_minor_ent_surgery": "minor surgery",
    "Surgery_they_are_scheduled_for_minor_breast_surgery": "minor surgery",
    "Surgery_they_are_scheduled_for_removal_of_a_pilonidal_cyst_on_their_back": "minor surgery",
    "Surgery_they_are_scheduled_for_a_whipple": "major surgery (not lower extremity)",
    "Surgery_they_are_scheduled_for_four_hour_arthroscopic_shoulder_surgery": "major surgery (not lower extremity)",
    "Surgery_they_are_scheduled_for_a_four_hour_laproscopic_nephrectomy": "major surgery (not lower extremity)",
    "Surgery_they_are_scheduled_for_elective_major_hip_arthroplasty": "major lower extremity surgery",
    "Surgery_they_are_scheduled_for_elective_major_total_hip_arthroplasty": "major lower extremity surgery",
    "Surgery_they_are_scheduled_for_elective_major_total_hip_replacement": "major lower extremity surgery",
}
# the template says "In the past one month, they have had no other significant
# medical events", so every recent-history field except the sampled one is false
CAPRINI_RECENT = {
    "Recent History_major_surgery": "recent_major_surgery",
    "Recent History_a_chf_exacerbation": "recent_chf",
    "Recent History_an_admission_for_sepsis": "recent_sepsis",
    "Recent History_an_admission_for_pneumonia": "recent_pna",
    "Recent History_a_radial_fracture_requiring_an_immobilizing_plaster_cast": "recent_cast",
    "Recent History_a_pelvic_fracture": "recent_lower_extremity_or_hip_or_pelvis_fracture",
    "Recent History_stroke_with_minimal_residual_deficits": "recent_stroke",
    "Recent History_multiple_traumatic_falls": "recent_trauma",
    "Recent History_acute_spinal_cord_injury_causing_paralysis": "recent_spinal_cord_injury",
}
CAPRINI_RECENT_FIELDS = sorted(set(CAPRINI_RECENT.values()) | {"recent_preg"})
# "They have no other venous or clotting disorders"
CAPRINI_VENOUS = {
    "Venous Disease_varicose_veins": "varicose_veins",
    "Venous Disease_current_swollen_legs": "current_swollen_legs",
    "Venous Disease_current_central_venous_access": "current_central_venous_access",
    "Venous Disease_history_of_dvtpe": "history_of_dvt_pe",
    "Venous Disease_positive_factor_v_leiden": "positive_factor_v_leiden",
    "Venous Disease_positive_prothrombin_20210a": "positive_prothrombin_20210a",
    "Venous Disease_elevated_serum_homocysteine": "elevated_serum_homocysteine",
    "Venous Disease_positive_lupus_anticoagulant": "positive_lupus_anticoagulant",
    "Venous Disease_elevated_anticardiolipin_antibody": "elevated_anticardiolipin_antibody",
    "Venous Disease_heparin_induced_thrombocytopenia": "heparin_induced_thrombocytopenia",
}
CAPRINI_VENOUS_FIELDS = sorted(
    set(CAPRINI_VENOUS.values())
    | {"family_history_of_thrombosis", "other_congenital_or_acquired_thrombophilia"}
)
CAPRINI_MOBILITY = {
    "Mobility_normal_they_are_ambulating_indepently": "normal/ambulatory",
    "Mobility_minimal_currently_as_they_are_on_medical_bedrest": "bedrest < 72 hours and/or only walking in room",
    "Mobility_very_poor_they_are_confined_to_bed_72_hours": "confined to bed >72 hours",
}
# "There is no other relevant past medical or surgical history"
CAPRINI_PMH = {
    "Other Pmh_history_of_inflammatory_bowel_disease": "history_of_inflammatory_bowel_disease",
    "Other Pmh_an_acute_mi_sustained_during_this_hospitalization": "acute_mi",
    "Other Pmh_history_of_copd": "copd",
    "Other Pmh_previous_malignancy": "present_or_previous_malignancy",
}
CAPRINI_PMH_FIELDS = sorted(set(CAPRINI_PMH.values()) | {"other_risk_factors"})


def caprini(options: List[Option]) -> Gold:
    b = _Builder(options)
    b.set("age", int(_num(b.opt("caprini_age").input_string)), "caprini_age")
    b.set("sex", {"man": "M", "woman": "F"}[b.opt("caprini_sex").input_string], "caprini_sex")
    b.set("type_of_surgery", _lookup(CAPRINI_SURGERY, b.opt("surgery").name, "surgery"), "surgery")
    for crit, table, fields in [
        ("recent_history", CAPRINI_RECENT, CAPRINI_RECENT_FIELDS),
        ("venous_disease", CAPRINI_VENOUS, CAPRINI_VENOUS_FIELDS),
        ("other_pmh", CAPRINI_PMH, CAPRINI_PMH_FIELDS),
    ]:
        positive = _lookup(table, b.opt(crit).name, crit)
        for f in fields:
            if f == positive:
                b.set(f, True, crit)
            else:
                b.set(f, False)
    b.set("mobility", _lookup(CAPRINI_MOBILITY, b.opt("mobility").name, "mobility"), "mobility")
    b.set("bmi", int(_num(b.opt("bmi").input_string)), "bmi")
    return b.gold()


# ---------------------------------------------------------------- Wells DVT

WELLS = {
    "active_cancer": "wells_cancer",
    "bedridden_or_major_surgery_recently": "wells_bedridden",
    "calf_swelling": "wells_swelling",
    "collateral_veins": "wells_collateral",
    "entire_leg_swollen": "wells_entire_leg",
    "localized_tenderness": "wells_tenderness",
    "pitting_edema": "wells_pitting",
    "paralysis_paresis_or_plaster": "wells_paralysis",
    "previous_dvt": "wells_prev_dvt",
    "alternative_diagnosis_as_likely": "wells_alternative",
}


def wellsdvt(options: List[Option]) -> Gold:
    b = _Builder(options)
    for field, crit in WELLS.items():
        b.flag(field, crit)
    return b.gold()


# ---------------------------------------------------------------- MELD-Na


def meldna(options: List[Option]) -> Gold:
    b = _Builder(options)
    for field in ["creatinine", "bilirubin", "inr", "sodium"]:
        b.set(field, _num(b.opt(field).input_string), field)
    b.set("is_on_dialysis", b.opt("is_on_dialysis").name == "Is_On_Dialysis_True", "is_on_dialysis")
    return b.gold()


# ---------------------------------------------------------------- SOFA

# respiratory option -> (mechanical ventilation, FiO2 %, PaO2), keyed by its points
SOFA_RESPIRATORY = {
    0: (False, 21.0, 220.0),
    1: (False, 30.0, 90.0),
    2: (False, 70.0, 58.0),
    3: (True, 70.0, 72.0),
    4: (True, 100.0, 65.0),
}
# cardiovascular option -> (MAP, dopamine, dobutamine, epinephrine, norepinephrine)
SOFA_CARDIO = {
    0: (73.0, 0.0, 0.0, 0.0, 0.0),
    1: (68.0, 0.0, 0.0, 0.0, 0.0),
    2: (72.0, 0.0, 3.0, 0.0, 0.0),
    3: (62.0, 0.0, 0.0, 0.0, 0.08),
    4: (60.0, 0.0, 0.0, 0.15, 0.15),
}


def _check_numbers_present(option: Option, *numbers):
    """Guard against the spreadsheet text drifting away from the tables above."""
    text = option.input_string
    for n in numbers:
        variants = {f"{n:g}", f"{n:.3f}", f"{n:.2f}", f"{n:.1f}"}
        if n and not any(v in text for v in variants):
            raise ValueError(f"{n} not found in option text {text!r}")


def sofa(options: List[Option]) -> Gold:
    b = _Builder(options)
    resp = b.opt("sofa_mechanical_ventilation")
    vent, fio2, pao2 = SOFA_RESPIRATORY[resp.score_effect]
    _check_numbers_present(resp, fio2, pao2)
    if vent != ("intubated" in resp.input_string):
        raise ValueError(f"ventilation table disagrees with {resp.input_string!r}")
    b.set("mechanical_ventilation", vent, "sofa_mechanical_ventilation")
    b.set("fio2", fio2, "sofa_mechanical_ventilation")
    b.set("pao2", pao2, "sofa_mechanical_ventilation")

    cv = b.opt("sofa_hypotension_and_or_pressors")
    values = SOFA_CARDIO[cv.score_effect]
    _check_numbers_present(cv, *values)
    for field, v in zip(
        ["map", "dopamine_dose", "dobutamine_dose", "epinephrine_dose", "norepinephrine_dose"], values
    ):
        b.set(field, v, "sofa_hypotension_and_or_pressors")

    b.set("platelets", int(_num(b.opt("sofa_platelets").input_string)), "sofa_platelets")
    b.set("glasgow_coma_scale", int(_num(b.opt("sofa_glasgow_coma_scale").input_string)), "sofa_glasgow_coma_scale")
    b.set("bilirubin", _num(b.opt("sofa_bilirubin").input_string), "sofa_bilirubin")
    b.set("creatinine", _num(b.opt("sofa_creatinine").input_string), "sofa_creatinine")
    b.set("urine_output", _num(b.opt("sofa_urine_output").input_string), "sofa_urine_output")
    return b.gold()


# ---------------------------------------------------------------- CCI

CCI_FLAGS = [
    "myocardial_infarction",
    "congestive_heart_failure",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "rheumatic_disease",
    "peptic_ulcer_disease",
    "hemiplegia_or_paraplegia",
    "renal_disease",
    "leukemia",
    "lymphoma",
    "aids_hiv",
]
CCI_LIVER = {0: "NO", 1: "YES_MILD", 3: "YES_MODERATE_SEVERE"}
CCI_DIABETES = {0: "NONE", 1: "UNCOMPLICATED", 2: "END_ORGAN_DAMAGE"}
CCI_TUMOR = {0: "NO", 2: "YES_LOCALIZED", 6: "YES_WITH_METASTASIS"}


def cci(options: List[Option]) -> Gold:
    b = _Builder(options)
    b.set("age", int(_num(b.opt("cci_age").input_string)), "cci_age")
    for field in CCI_FLAGS:
        b.flag(field, f"cci_{field}")
    b.set("liver_disease", CCI_LIVER[b.opt("cci_liver_disease").score_effect], "cci_liver_disease")
    b.set("diabetes", CCI_DIABETES[b.opt("cci_diabetes").score_effect], "cci_diabetes")
    b.set("solid_tumor", CCI_TUMOR[b.opt("cci_solid_tumor").score_effect], "cci_solid_tumor")
    return b.gold()


# ---------------------------------------------------------------- ARISCAT

ARISCAT_INCISION = {
    "ARISCAT Peripheral Incision": "Peripheral",
    "ARISCAT Upper Abdominal Incision": "Upper abdominal",
    "ARISCAT Intrathoracic Incision": "Intrathoracic",
}
ARISCAT_DURATION_HOURS = {"one hour": 1.0, "2.5 hours": 2.5, "4 hours": 4.0}


def ariscat(options: List[Option]) -> Gold:
    b = _Builder(options)
    b.set("age", int(_num(b.opt("ariscat_age").input_string)), "ariscat_age")
    b.set("preoperative_spo2", int(_num(b.opt("ariscat_spo2").input_string)), "ariscat_spo2")
    b.flag("respiratory_infection_with_fever_and_antibiotics", "ariscat_respiratory_infection")
    b.flag("preoperative_anemia", "ariscat_anemia")
    b.set(
        "surgical_incision",
        _lookup(ARISCAT_INCISION, b.opt("ariscat_surgical_incision").name, "incision"),
        "ariscat_surgical_incision",
    )
    b.set(
        "duration_of_surgery",
        _lookup(ARISCAT_DURATION_HOURS, b.opt("ariscat_duration_of_surgery").input_string, "duration"),
        "ariscat_duration_of_surgery",
    )
    b.flag("emergency_procedure", "ariscat_emergency_procedure")
    return b.gold()


# ---------------------------------------------------------------- PSI/PORT

PSI_FLAGS = [
    "nursing_home_resident",
    "neoplastic_disease",
    "liver_disease",
    "congestive_heart_failure",
    "cerebrovascular_disease",
    "renal_disease",
    "altered_mental_status",
]
# schema field -> (criterion, type)
PSI_VALUES = {
    "respiratory_rate": ("psiport_respiratory_rate", int),
    "systolic_bp": ("psiport_systolic_bp", int),
    "pulse": ("psiport_pulse", int),
    "temperature": ("psiport_temperature", float),
    "ph": ("psiport_ph", float),
    "bun": ("psiport_bun", int),
    "sodium_mmol_L": ("psiport_sodium", int),
    "glucose": ("psiport_glucose", int),
    "hematocrit": ("psiport_hematocrit", int),
    "pao2": ("psiport_pao2", int),
}


def psiport(options: List[Option]) -> Gold:
    b = _Builder(options)
    b.set("age", int(_num(b.opt("psiport_age").input_string)), "psiport_age")
    b.set("sex", {"male": "M", "female": "F"}[b.opt("psiport_sex").input_string], "psiport_sex")
    for field in PSI_FLAGS:
        b.flag(field, f"psiport_{field}")
    for field, (crit, cast) in PSI_VALUES.items():
        b.set(field, cast(_num(b.opt(crit).input_string)), crit)
    # the schema types pleural_effusion as int; the calculator only tests truthiness
    b.set("pleural_effusion", int(b.opt("psiport_pleural_effusion").score_effect != 0), "psiport_pleural_effusion")
    return b.gold()


# ---------------------------------------------------------------- GAD-7

GAD7_FREQUENCY = {0: "NOT_AT_ALL", 1: "SEVERAL_DAYS", 2: "MORE_THAN_HALF_THE_DAYS", 3: "NEARLY_EVERY_DAY"}
GAD7_SINGLE = ["feeling_nervous", "trouble_relaxing", "restlessness", "easily_annoyed", "feeling_afraid"]
# the difficulty option names reuse the frequency labels; map them onto the difficulty enum
GAD7_DIFFICULTY = {
    "GAD7 Difficulty (Not at all)": "NOT_AT_ALL",
    "GAD7 Difficulty (Several days)": "SOMEWHAT_DIFFICULT",
    "GAD7 Difficulty (More than half the days)": "VERY_DIFFICULT",
    "GAD7 Difficulty (Nearly every day)": "EXTREMELY_DIFFICULT",
}


def gad7(options: List[Option]) -> Gold:
    b = _Builder(options)
    for field in GAD7_SINGLE:
        b.set(field, GAD7_FREQUENCY[b.opt(f"gad7_{field}").score_effect], f"gad7_{field}")
    # one option covers both worry items, e.g. "GAD7 Worry Two Criteria - 3 + 2";
    # the gad7_worrying_too_much placeholder only inserts a distractor sentence
    worry = b.opt("gad7_cant_stop_worrying")
    m = re.search(r"(\d) \+ (\d)", worry.name)
    if not m:
        raise ValueError(f"cannot parse worry option {worry.name!r}")
    b.set("cant_stop_worrying", GAD7_FREQUENCY[int(m.group(1))], "gad7_cant_stop_worrying")
    b.set("worrying_too_much", GAD7_FREQUENCY[int(m.group(2))], "gad7_cant_stop_worrying")
    b.set("difficulty", _lookup(GAD7_DIFFICULTY, b.opt("gad7_difficulty").name, "difficulty"), "gad7_difficulty")
    return b.gold()


# ---------------------------------------------------------------- HAS-BLED

HASBLED_FLAGS = {
    "hypertension": "hasbled_hypertension",
    "severe_renal_disease": "hasbled_renal_disease",
    "severe_liver_disease": "hasbled_liver_disease",
    "stroke_history": "hasbled_stroke_history",
    "prior_major_bleeding": "hasbled_prior_major_bleeding",
    "labile_inr": "hasbled_labile_inr",
    "drugs": "hasbled_drugs",
    "alcohol": "hasbled_alcohol",
}


def hasbled(options: List[Option]) -> Gold:
    b = _Builder(options)
    b.set("age", int(_num(b.opt("hasbled_age").input_string)), "hasbled_age")
    for field, crit in HASBLED_FLAGS.items():
        b.flag(field, crit)
    return b.gold()


# ---------------------------------------------------------------- NIHSS

NIHSS_FIELD = {
    "nihss_consciousness": "consciousness",
    "nihss_questions": "questions",
    "nihss_commands": "commands",
    "nihss_gaze": "gaze",
    "nihss_visual_fields": "visual",
    "nihss_facial_palsy": "facial_palsy",
    "nihss_motor_arm_left": "motor_arm_left",
    "nihss_motor_arm_right": "motor_arm_right",
    "nihss_motor_leg_left": "motor_leg_left",
    "nihss_motor_leg_right": "motor_leg_right",
    "nihss_ataxia": "limb_ataxia",
    "nihss_sensory": "sensory",
    "nihss_language": "language",
    "nihss_dysarthria": "dysarthria",
    "nihss_extinction": "extinction",
}


def nihss(options: List[Option]) -> Gold:
    b = _Builder(options)
    for crit, field in NIHSS_FIELD.items():
        # option names are literally '"motor_arm_left": "NO_DRIFT"'
        m = re.fullmatch(r'"(\w+)": "(\w+)"', b.opt(crit).name.strip())
        if not m or m.group(1) != field:
            raise ValueError(f"unexpected NIHSS option name {b.opt(crit).name!r} for {field}")
        b.set(field, m.group(2), crit)
    return b.gold()


MAPPERS: Dict[str, Callable[[List[Option]], Gold]] = {
    "caprini": caprini,
    "wellsdvt": wellsdvt,
    "meldna": meldna,
    "sofa": sofa,
    "cci": cci,
    "ariscat": ariscat,
    "psiport": psiport,
    "gad7": gad7,
    "hasbled": hasbled,
    "nihss": nihss,
}


def gold_for(case: GeneratedCase) -> Gold:
    gold = MAPPERS[case.calculator](case.options)
    model = omc.request_model(case.calculator)
    # validation catches wrong enum values and missing or extra fields
    validated = model(**gold.request)
    missing = set(model.model_fields) - set(gold.request)
    if missing:
        raise ValueError(f"{case.calculator}: gold request missing {missing}")
    return Gold(validated.model_dump(mode="json"), gold.sources)


def score(calculator: str, request: Dict) -> int:
    return omc.calculate(calculator, omc.request_model(calculator)(**request))


def round_trip(case: GeneratedCase) -> Tuple[bool, int, int]:
    """(agrees, OpenMedCalc score of the gold request, original ground truth)."""
    computed = score(case.calculator, gold_for(case).request)
    original = int(float(case.original_score))
    return computed == original, computed, original
