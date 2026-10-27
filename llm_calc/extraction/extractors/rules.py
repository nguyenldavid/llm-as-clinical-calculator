"""
System A: rule-based clinical NLP (no learned components).

Building blocks, in the spirit of medspaCy/ConText but dependency-free:

* sentences and clauses split on punctuation and coordinating "but/and/or";
* concept patterns with clinical synonyms and abbreviations (CVA, h/o, HD ...);
* negation: a cue (no, denies, without, ...) earlier in the same clause;
* attribution: sentences about relatives (father, mother, family) are skipped;
* numbers anchored to a label ("creatinine is 1.3", "MAP 62"), with unit
  conversion shared with the encoder (``units.normalise``);
* thresholds applied where the decision schema folds them into a boolean.

Caveat for interpreting results: the rules were written by someone who had read
the T1 option phrasings, so T1 and A409 scores are in-sample for this system.
T2-T4 are the fair comparison.
"""

import re
from typing import Callable, Dict, List, Optional

from llm_calc.extraction import units
from llm_calc.extraction.extractors.base import Extractor, Timer, prediction

NEGATION = re.compile(
    r"\b(no|not|denies|deny|denied|without|negative for|none|absence of|abstinence|abstains|free of|never|"
    r"haven't|hasn't|didn't|did not|does not|doesn't|nor|unremarkable|normal)\b",
    re.IGNORECASE,
)
FAMILY = re.compile(r"\b(father|mother|brother|sister|sibling|family history|grandparent|son|daughter|aunt|uncle)\b",
                    re.IGNORECASE)


def sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def clauses(sentence: str) -> List[str]:
    # negation carries across lists ("no paralysis, paresis, or plaster"), so only
    # contrastive connectives end its scope
    return [c for c in re.split(r";|\bbut\b|\bhowever\b|\balthough\b|\bthough\b", sentence) if c.strip()]


def _patient_sentences(text: str) -> List[str]:
    """Sentences about the patient: not about relatives, and not the closing question."""
    return [s for s in sentences(text) if not FAMILY.search(s) and not s.rstrip().endswith("?")]


def mentioned(text: str, concept: str) -> Optional[bool]:
    """True if asserted, False if only negated, None if never mentioned."""
    pattern = re.compile(concept, re.IGNORECASE)
    seen_negated = False
    for s in _patient_sentences(text):
        for c in clauses(s):
            for m in pattern.finditer(c):
                if NEGATION.search(c[: m.start()]):
                    seen_negated = True
                else:
                    return True
    return False if seen_negated else None


def flag(text: str, concept: str) -> bool:
    return bool(mentioned(text, concept))


def number_after(text: str, label: str, field: str = "", window: int = 25) -> Optional[float]:
    """First number within ``window`` characters after ``label`` (unit-normalised)."""
    for s in _patient_sentences(text):
        for m in re.finditer(label, s, re.IGNORECASE):
            tail = s[m.end(): m.end() + window]
            n = units.NUMBER.search(tail)
            if n:
                value = units.number_value(n.group())
                return units.normalise(field or label, value, tail[n.end():])
    return None


def first_match(text: str, options: List[tuple], default):
    """``options`` is [(pattern, value)], checked in order against patient sentences."""
    body = " ".join(_patient_sentences(text))
    for pattern, value in options:
        if re.search(pattern, body, re.IGNORECASE):
            return value
    return default


def _age(text: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\s*(?:-?\s*year[- ]old|yo\b|y/o|years old|-yo)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    v = number_after(text, r"\bage[d]?\b", window=8)
    return int(v) if v is not None else None


def _sex(text: str) -> str:
    if re.search(r"\b(woman|female|she|her)\b", text, re.IGNORECASE):
        return "F"
    return "M"


# ------------------------------------------------------------------ calculators


def caprini(t: str, variant: str) -> Dict:
    surgery = "not scheduled for surgery"
    if not re.search(r"not scheduled for (any )?surgery", t, re.IGNORECASE) and re.search(
            r"scheduled for|undergo|planned", t, re.IGNORECASE):
        minutes = number_after(t, r"scheduled for an?", window=6)
        if re.search(r"\b(hip|knee)\b.*\b(arthroplasty|replacement)|lower extremity", t, re.IGNORECASE):
            surgery = "major lower extremity surgery"
        elif re.search(r"local anesthesia|\bminor\b|-minute", t, re.IGNORECASE) and not re.search(
                r"\bhour", t, re.IGNORECASE):
            surgery = "minor surgery"
        else:
            surgery = "major surgery (not lower extremity)"
        del minutes
    mobility = first_match(t, [
        (r"confined to bed|bed-?bound|bedridden|bed for (three|several|\d+) weeks", "confined to bed >72 hours"),
        (r"bed ?rest|walk(ing)? (around )?(their|the) room", "bedrest < 72 hours and/or only walking in room"),
    ], "normal/ambulatory")
    bmi = number_after(t, r"\bBMI\b( is| of)?")
    return {
        "age": _age(t) or 0,
        "sex": _sex(t),
        "type_of_surgery": surgery,
        "recent_major_surgery": flag(t, r"major surgery at|recent major surgery|major surgery (two|three|\d+) weeks"),
        "recent_chf": flag(t, r"\bCHF\b|heart failure"),
        "recent_sepsis": flag(t, r"sepsis|septic"),
        "recent_pna": flag(t, r"pneumonia|\bPNA\b"),
        "recent_preg": flag(t, r"pregnan|postpartum"),
        "recent_cast": flag(t, r"plaster cast|\bcast\b|casted"),
        "recent_lower_extremity_or_hip_or_pelvis_fracture": flag(t, r"(pelvic|pelvis|hip|femur|femoral|tibia|leg) fracture|fracture of the (pelvis|hip|femur|leg)"),
        "recent_stroke": flag(t, r"stroke|\bCVA\b"),
        "recent_trauma": flag(t, r"trauma|traumatic|falls\b"),
        "recent_spinal_cord_injury": flag(t, r"spinal cord"),
        "varicose_veins": flag(t, r"varicose"),
        "current_swollen_legs": flag(t, r"swollen legs|leg swelling|swelling of (the|their) legs"),
        "current_central_venous_access": flag(t, r"central venous|central line|\bPICC\b|port-a-cath"),
        "history_of_dvt_pe": flag(t, r"\bDVT\b|\bPE\b|pulmonary embol|\bVTE\b|venous thromboembol"),
        "family_history_of_thrombosis": False,  # family sentences are skipped by design; see FAMILY
        "positive_factor_v_leiden": flag(t, r"factor v leiden"),
        "positive_prothrombin_20210a": flag(t, r"prothrombin 20210"),
        "elevated_serum_homocysteine": flag(t, r"homocysteine"),
        "positive_lupus_anticoagulant": flag(t, r"lupus anticoagulant"),
        "elevated_anticardiolipin_antibody": flag(t, r"anticardiolipin"),
        "heparin_induced_thrombocytopenia": flag(t, r"heparin[- ]induced|\bHIT\b"),
        "other_congenital_or_acquired_thrombophilia": flag(t, r"thrombophilia"),
        "mobility": mobility,
        "history_of_inflammatory_bowel_disease": flag(t, r"inflammatory bowel|\bIBD\b|crohn|ulcerative colitis"),
        "bmi": int(bmi) if bmi is not None else 0,
        "acute_mi": flag(t, r"acute MI|myocardial infarction|\bSTEMI\b|\bNSTEMI\b"),
        "copd": flag(t, r"\bCOPD\b|emphysema|chronic obstructive"),
        "present_or_previous_malignancy": flag(t, r"malignan|cancer|carcinoma|tumou?r"),
        "other_risk_factors": False,
    }


def wellsdvt(t: str, variant: str) -> Dict:
    alt = mentioned(t, r"alternative diagnosis")
    return {
        "active_cancer": flag(t, r"cancer|malignan|neoplas"),
        "bedridden_or_major_surgery_recently": flag(t, r"bedridden|bed-?bound|bed ?rest|major surgery|recent surgery"),
        "calf_swelling": flag(t, r"calf swelling|swelling of the (left |right )?calf|calf .{0,20}\d+\s*cm"),
        "collateral_veins": flag(t, r"collateral"),
        "entire_leg_swollen": flag(t, r"entire leg|whole leg"),
        "localized_tenderness": flag(t, r"tender"),
        "pitting_edema": flag(t, r"pitting"),
        "paralysis_paresis_or_plaster": flag(t, r"paraly|paresis|weakness|plaster|\bcast(ed)?\b|immobiliz"),
        "previous_dvt": flag(t, r"(prior|previous|history of|h/o)[^.]{0,30}\bDVT\b|\bDVT\b[^.]{0,20}(previously|before)"),
        "alternative_diagnosis_as_likely": bool(alt) and bool(re.search(r"as likely|more likely|equally likely", t, re.IGNORECASE)),
    }


def meldna(t: str, variant: str) -> Dict:
    return {
        "is_on_dialysis": flag(t, r"dialysis|\bHD\b|hemodialysis|\bCRRT\b"),
        "creatinine": number_after(t, r"creatinine( is| of|:)?", "creatinine") or 1.0,
        "bilirubin": number_after(t, r"bilirubin( is| of|:)?", "bilirubin") or 1.0,
        "inr": number_after(t, r"\bINR\b( is| of|:)?") or 1.0,
        "sodium": number_after(t, r"sodium( is| of|:)?|\bNa\b") or 137.0,
    }


def _dose(t: str, drug: str) -> float:
    for s in _patient_sentences(t):
        if re.search(drug, s, re.IGNORECASE):
            m = re.search(r"(\d+(?:\.\d+)?)\s*(mcg|µg|ug)/kg/min", s, re.IGNORECASE)
            if m:
                return float(m.group(1))
    return 0.0


def sofa(t: str, variant: str) -> Dict:
    vent = flag(t, r"intubated|mechanical(ly)? ventilat|\bETT\b|\bVC/?AC\b|BiPAP|\bCPAP\b") and not re.search(
        r"spontaneously without assistance", t, re.IGNORECASE)
    return {
        "pao2": number_after(t, r"PaO2( is| of|:)?") or 100.0,
        "fio2": number_after(t, r"FiO2( is| of|:)?") or 21.0,
        "mechanical_ventilation": vent,
        "platelets": int(number_after(t, r"platelet(s| count)?( are| is| of|:)?") or 200),
        "glasgow_coma_scale": int(number_after(t, r"Glasgow Coma Scale( score)?( is| of|:)?|\bGCS\b") or 15),
        "bilirubin": number_after(t, r"bilirubin( level)?( is| of|:)?", "bilirubin") or 0.5,
        "map": number_after(t, r"\bMAP\b( is| of|:)?|mean arterial pressure( is| of)?") or 75.0,
        "dopamine_dose": _dose(t, r"dopamine|\bdopa\b|intropin"),
        "dobutamine_dose": _dose(t, r"dobutamine|dobutrex"),
        "epinephrine_dose": _dose(t, r"(?<!nor)(?<!nor-)epinephrine|(?<!nor)\bepi\b|adrenaline"),
        "norepinephrine_dose": _dose(t, r"norepinephrine|norepi|levophed|\blevo\b|noradrenaline"),
        "creatinine": number_after(t, r"creatinine( level)?( is| of|:)?", "creatinine") or 0.8,
        "urine_output": number_after(t, r"urine output( of| is|:)?") or 1500.0,
    }


def cci(t: str, variant: str) -> Dict:
    liver = "NO"
    if flag(t, r"portal hypertension|varice|variceal|ascites|hepatic encephalopathy"):
        liver = "YES_MODERATE_SEVERE"
    elif flag(t, r"hepatitis|cirrhosis|fatty liver|liver disease|NAFLD|NASH"):
        liver = "YES_MILD"
    diabetes = "NONE"
    if flag(t, r"diabet|\bT[12]DM\b|\bDM\b"):
        diabetes = "END_ORGAN_DAMAGE" if re.search(r"retinopathy|neuropathy|nephropathy|end-organ", t, re.IGNORECASE) else "UNCOMPLICATED"
    tumor = "NO"
    if flag(t, r"(?<!no solid )\b(cancer|carcinoma|solid tumou?r|sarcoma|melanoma)"):
        metastatic = re.search(r"metastas[ie]s to|metastatic|stage (4|iv)\b", t, re.IGNORECASE) and not re.search(
            r"without (known )?metasta", t, re.IGNORECASE)
        tumor = "YES_WITH_METASTASIS" if metastatic else "YES_LOCALIZED"
    return {
        "age": _age(t) or 0,
        "myocardial_infarction": flag(t, r"myocardial infarction|heart attack|\bMI\b|\bSTEMI\b"),
        "congestive_heart_failure": flag(t, r"heart failure|\bCHF\b|\bHFrEF\b|\bHFpEF\b"),
        "peripheral_vascular_disease": flag(t, r"peripheral (vascular|arter\w*) disease|claudication|\bPAD\b|\bPVD\b"),
        "cerebrovascular_disease": flag(t, r"stroke|transient ischemic|\bTIA\b|\bCVA\b"),
        "dementia": flag(t, r"dementia|alzheimer|cognitive impairment"),
        "chronic_pulmonary_disease": flag(t, r"emphysema|\bCOPD\b|chronic (lung|pulmonary)|asthma|pulmonary fibrosis"),
        "rheumatic_disease": flag(t, r"rheumat|lupus|\bSLE\b|connective tissue|scleroderma"),
        "peptic_ulcer_disease": flag(t, r"peptic ulcer|ulcers\b|\bPUD\b"),
        "liver_disease": liver,
        "diabetes": diabetes,
        "hemiplegia_or_paraplegia": flag(t, r"hemipleg|parapleg|quadripleg|tetrapleg"),
        "renal_disease": flag(t, r"kidney disease|renal disease|\bCKD\b|\bESRD\b|dialysis"),
        "solid_tumor": tumor,
        "leukemia": flag(t, r"leuka?emia|\bAML\b|\bCML\b|\bCLL\b"),
        "lymphoma": flag(t, r"lymphoma"),
        "aids_hiv": flag(t, r"\bAIDS\b"),
    }


def ariscat(t: str, variant: str) -> Dict:
    hgb = number_after(t, r"\b(hgb|hemoglobin|hb)\b( is| of|:)?")
    hct = number_after(t, r"\b(hct|hematocrit)\b( is| of|:)?")
    anemia = (hgb is not None and hgb <= 10) or (hct is not None and hct < 30) or (
        hgb is None and hct is None and flag(t, r"anemi"))
    infection = flag(t, r"pneumonia|bronchitis|respiratory (tract )?infection|\bURI\b|febrile") and bool(
        re.search(r"antibiotic|\w+mycin\b|\w+cillin\b|\w+floxacin\b|\bcef\w+|doxycycline|augmentin|bactrim", t, re.IGNORECASE))
    incision = first_match(t, [
        (r"\bCABG\b|thoracotomy|lobectomy|esophagectomy|intrathoracic|cardiac surgery|\bVATS\b|sternotomy", "Intrathoracic"),
        (r"cholecystectomy|gastrectomy|upper abdominal|hepatectomy|whipple|pancreat|splenectomy", "Upper abdominal"),
    ], "Peripheral")
    hours = number_after(t, r"duration( of)?", window=12)
    if hours is not None and re.search(r"duration( of)?\s*\S+\s*min", t, re.IGNORECASE):
        hours = hours / 60
    return {
        "age": _age(t) or 0,
        "preoperative_spo2": int(number_after(t, r"SpO2( of| is|:)?|saturation( of| is)?") or 98),
        "respiratory_infection_with_fever_and_antibiotics": infection,
        "preoperative_anemia": bool(anemia),
        "surgical_incision": incision,
        "duration_of_surgery": hours if hours is not None else 1.0,
        "emergency_procedure": flag(t, r"emergen|\burgent"),
    }


def psiport(t: str, variant: str) -> Dict:
    return {
        "age": _age(t) or 0,
        "sex": _sex(t) if not re.search(r"\bmale\b", t, re.IGNORECASE) else "M",
        "nursing_home_resident": flag(t, r"nursing (home|facility)|skilled nursing|long-term care|assisted living"),
        "neoplastic_disease": flag(t, r"cancer|neoplas|malignan|carcinoma|tumou?r"),
        "liver_disease": flag(t, r"cirrhosis|hepatitis|liver disease"),
        "congestive_heart_failure": flag(t, r"heart failure|\bCHF\b|\bHFrEF\b"),
        "cerebrovascular_disease": flag(t, r"stroke|transient ischemic|\bTIA\b|cerebrovascular|\bCVA\b"),
        "renal_disease": flag(t, r"renal disease|kidney disease|\bESRD\b|\bCKD\b|dialysis"),
        "altered_mental_status": flag(t, r"confus|disorient|altered mental|obtunded|letharg|delirium"),
        "respiratory_rate": int(number_after(t, r"respiratory rate( of| is|:)?|\bRR\b") or 16),
        "systolic_bp": int(number_after(t, r"systolic (BP|blood pressure)( of| is|:)?|\bSBP\b") or 120),
        "temperature": number_after(t, r"temperature( of| is|:)?|\btemp\b", "temperature") or 37.0,
        "pulse": int(number_after(t, r"pulse( of| is|:)?|heart rate( of| is)?|\bHR\b") or 80),
        "ph": number_after(t, r"\bpH\b( of| is|:)?") or 7.4,
        "bun": int(number_after(t, r"\bBUN\b( of| is|:)?|blood urea nitrogen") or 15),
        "sodium_mmol_L": int(number_after(t, r"sodium( of| is|:)?|\bNa\b") or 140),
        "glucose": int(round(number_after(t, r"glucose( of| is|:)?", "glucose") or 100)),
        "hematocrit": int(number_after(t, r"hematocrit( of| is|:)?|\bhct\b") or 40),
        "pao2": int(number_after(t, r"PaO2( of| is|:)?") or 90),
        "pleural_effusion": int(flag(t, r"effusion")),
    }


GAD7_FREQ = [
    (r"nearly (every|all the) day|every day|nearly every", "NEARLY_EVERY_DAY"),
    (r"more than half", "MORE_THAN_HALF_THE_DAYS"),
    (r"several days|some days|a few days", "SEVERAL_DAYS"),
    (r"not at all|haven't|have not|did not|didn't|no |never|any day", "NOT_AT_ALL"),
]
GAD7_ITEMS = {
    "feeling_nervous": r"nervous|anxious|on edge",
    "cant_stop_worrying": r"control (in |over |of )?(their )?worr|stop or control worr|can't stop worr|unable to stop worr",
    "worrying_too_much": r"worrying too much",
    "trouble_relaxing": r"relax",
    "restlessness": r"restless",
    "easily_annoyed": r"annoyed|irritab",
    "feeling_afraid": r"afraid|fear",
}


def gad7(t: str, variant: str) -> Dict:
    out = {}
    for field, concept in GAD7_ITEMS.items():
        value = "NOT_AT_ALL"
        for s in sentences(t):
            parts = re.split(r",\s*(?=but|and|or)\b", s)
            for i, c in enumerate(parts):
                if re.search(concept, c, re.IGNORECASE):
                    # "haven't experienced X, or found Y" carries the negation into the next clause
                    scope = c if i == 0 else parts[i - 1] + c if c.strip().startswith("or") else c
                    value = next((v for p, v in GAD7_FREQ if re.search(p, scope, re.IGNORECASE)), "NOT_AT_ALL")
                    break
            else:
                continue
            break
        out[field] = value
    out["difficulty"] = first_match(t, [
        (r"extremely difficult", "EXTREMELY_DIFFICULT"),
        (r"very difficult", "VERY_DIFFICULT"),
        (r"somewhat difficult", "SOMEWHAT_DIFFICULT"),
    ], "NOT_AT_ALL")
    return out


def _hasbled_facts(t: str) -> Dict:
    sbp = None
    m = re.search(r"(?:blood pressure|\bBP\b)\D{0,10}(\d{2,3})\s*/\s*\d{2,3}", t, re.IGNORECASE)
    if m:
        sbp = int(m.group(1))
    drinks = None
    for s in _patient_sentences(t):
        if re.search(r"alcohol|drinks?|\bEtOH\b", s, re.IGNORECASE):
            if re.search(r"abstinen|abstain|\bno (EtOH|alcohol)\b|does not drink|doesn't drink|denies (alcohol|drinking)", s, re.IGNORECASE):
                drinks = 0.0
                break
            n = units.NUMBER.search(s)
            if n:
                drinks = units.number_value(n.group())
                break
    return {
        "age_years": _age(t),
        "systolic_bp": sbp,
        "creatinine_mg_dl": number_after(t, r"creatinine( is| of|:)?", "creatinine"),
        "on_dialysis": mentioned(t, r"dialysis|\bHD\b"),
        "kidney_transplant": mentioned(t, r"(kidney|renal) transplant"),
        "cirrhosis": mentioned(t, r"cirrhosis|liver disease"),
        "stroke_history": mentioned(t, r"stroke|\bCVA\b"),
        "prior_major_bleeding": mentioned(t, r"bleed|\bGIB\b|hemorrhag|haemorrhag|transfus|\bpRBCs?\b"),
        "labile_inr": True if re.search(r"labile INR|unstable INR|fluctuating INR|TTR\s*<\s*60", t, re.IGNORECASE)
        else (False if re.search(r"stable INR", t, re.IGNORECASE) else None),
        "antiplatelet_or_nsaid_use": mentioned(
            t, r"aspirin|clopidogrel|\bNSAIDs?\b|ibuprofen|naproxen|antiplatelet|ticagrelor|prasugrel|"
               r"drugs that (might |may |could )?(increase|raise) (the )?bleeding"),
        "alcoholic_drinks_per_week": drinks,
    }


def hasbled(t: str, variant: str) -> Dict:
    facts = _hasbled_facts(t)
    if variant == "R":
        return facts
    from llm_calc.extraction.schemas_raw import RawRequestHASBLED

    decision = RawRequestHASBLED(**facts).to_decision()
    # the decision schema also accepts a stated diagnosis of uncontrolled hypertension
    if re.search(r"uncontrolled hypertension", t, re.IGNORECASE):
        decision["hypertension"] = True
    return decision


NIHSS_RULES: Dict[str, tuple] = {
    "consciousness": (r"\balert\b|repeated stimulation|responsive|\bcoma\b|level of consciousness", [
        (r"\bcoma|unresponsive", "COMA"), (r"(?<!not )(?<!n't )requires? (repeated or )?painful|only to pain", "OBTUNDED"),
        (r"not alert|repeated stimulation|drowsy", "NOT_ALERT")], "ALERT"),
    "questions": (r"question|orientation|age and the current month", [
        (r"neither|unable to (provide|answer)|either of the", "ANSWERS_NEITHER"),
        (r"only (answer )?one|one of the two", "ANSWERS_ONE")], "ANSWERS_BOTH"),
    "commands": (r"command", [
        (r"cannot perform either|neither|unable to perform", "PERFORMS_NEITHER"),
        (r"executes one|one command|only one", "PERFORMS_ONE")], "PERFORMS_BOTH"),
    "gaze": (r"\bgaze\b|eye movements?|deviation of the eyes|forced deviation", [
        (r"forced deviation|unable to move them past", "FORCED_DEVIATION"),
        (r"partial gaze|difficulty looking", "PARTIAL_GAZE_PALSY")], "NORMAL"),
    "visual": (r"visual|hemianopia|vision", [
        (r"bilateral", "BILATERAL_HEMIANOPIA"),
        (r"complete (absence of vision|hemianopia)|complete loss of vision", "COMPLETE_HEMIANOPIA"),
        (r"quadrant|partial hemianopia", "PARTIAL_HEMIANOPIA")], "NO_VISUAL_LOSS"),
    "facial_palsy": (r"facial|face|smile", [
        (r"complete paralysis|no movement", "COMPLETE"),
        (r"marked asymmetry|diminished|partial", "PARTIAL"),
        (r"minor asymmetry|minor|flattening", "MINOR")], "NORMAL"),
    "limb_ataxia": (r"ataxi|coordination|finger-nose|heel-shin|FNF", [
        (r"no ataxia|w/o ataxia|without ataxia", "ABSENT"),
        (r"both (legs|arms|limbs)|two limbs", "PRESENT_IN_TWO_LIMBS"),
        (r"one (leg|arm|limb)", "PRESENT_IN_ONE_LIMB")], "ABSENT"),
    "sensory": (r"sensation|pinprick|sensory", [
        (r"severe|total loss|unable to feel", "SEVERE_TO_TOTAL_LOSS"),
        (r"mild|moderate|diminished|altered feeling", "MILD_TO_MODERATE_LOSS")], "NORMAL"),
    "language": (r"language|aphasi|communicat|name objects|comprehension", [
        (r"\bmute\b|global aphasia", "MUTE"), (r"severe aphasia", "SEVERE_APHASIA"),
        (r"some difficulty|struggling|mild|moderate", "MILD_TO_MODERATE_APHASIA")], "NO_APHASIA"),
    "dysarthria": (r"speech|dysarthri|articulat", [
        (r"\bunintelligible|severe", "SEVERE"), (r"slurred|mild|moderate", "MILD_TO_MODERATE")], "NORMAL"),
    "extinction": (r"neglect|inattention|extinction|simultaneous stimuli", [
        (r"profound|do not recognize their own hand|does not recognize", "PROFOUND_HEMI_INATTENTION"),
        (r"fails to acknowledge|extinction to|inattention to one", "VISUAL_TACTILE_AUDITORY_SPATIAL_OR_PERSONAL_INATTENTION")],
        "NO_ABNORMALITY"),
}
MOTOR = [
    (r"no (observable )?voluntary movement|unable to produce any voluntary movement|no movement", "NO_MOVEMENT"),
    (r"falls immediately|no effort against gravity", "NO_EFFORT_AGAINST_GRAVITY"),
    (r"against gravity but|some effort against gravity|cannot maintain", "SOME_EFFORT_AGAINST_GRAVITY"),
    (r"without drifting|no drift|maintains its position", "NO_DRIFT"),
    (r"drift", "DRIFT"),
]


def nihss(t: str, variant: str) -> Dict:
    out = {}
    sents = sentences(t)
    for field, (topic, patterns, default) in NIHSS_RULES.items():
        value = default
        for s in sents:
            if re.search(topic, s, re.IGNORECASE):
                value = next((v for p, v in patterns if re.search(p, s, re.IGNORECASE)), default)
                break
        out[field] = value
    for side in ("left", "right"):
        for limb in ("arm", "leg"):
            value = "NO_DRIFT"
            for s in sents:
                if re.search(rf"\b{side} {limb}\b", s, re.IGNORECASE):
                    value = next((v for p, v in MOTOR if re.search(p, s, re.IGNORECASE)), "NO_DRIFT")
                    break
            out[f"motor_{limb}_{side}"] = value
    return out


RULES: Dict[str, Callable[[str, str], Dict]] = {
    "caprini": caprini, "wellsdvt": wellsdvt, "meldna": meldna, "sofa": sofa, "cci": cci,
    "ariscat": ariscat, "psiport": psiport, "gad7": gad7, "hasbled": hasbled, "nihss": nihss,
}


class RulesExtractor(Extractor):
    name = "rules"

    def extract_batch(self, records, variant):
        out = []
        for r in records:
            with Timer() as t:
                try:
                    request, error = RULES[r["calculator"]](r["vignette"], variant), None
                except Exception as e:  # noqa: BLE001 - a rule crash is a failed prediction
                    request, error = None, f"rules failed: {e!r}"
            out.append(prediction(r, self.name, variant, request, error=error, latency_s=t.elapsed))
        return out
