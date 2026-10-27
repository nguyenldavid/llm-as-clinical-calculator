"""
T4: controlled perturbations of generated cases that leave the gold request unchanged.

Each perturbation re-renders a case from its template and sampled options, so it
is applied at the level of option text rather than by editing finished prose:

* ``distractors``  irrelevant sentences, including a family-history sentence that
                   names conditions the calculators score (attribution test)
* ``units``        SI units for labs and Fahrenheit for temperature (the schema
                   stays in US units, so a correct extractor must convert)
* ``conflicting``  an older, different value next to each current lab value
* ``abbreviations`` clinical shorthand for option phrasings that have one
* ``reorder``      the option sentences in random order (defeats position cues)
* ``long_note``    the case embedded in a longer note with unrelated sections

Not every perturbation applies to every calculator (units only where labs are
given); ``perturb`` returns None when it would leave the text unchanged.
"""

import random
import re
from typing import Callable, Dict, List, Optional

from llm_calc.extraction import generator as g
from llm_calc.extraction.extractors.baselines import _template_parts

DISTRACTORS = [
    "Their daughter accompanied them to today's visit",
    "They work as a high school teacher and enjoy hiking on weekends",
    "They received their seasonal influenza vaccine last month",
    "They report a penicillin allergy that causes a rash",
    "Their father had a stroke at age 80 and their mother had liver cirrhosis",
    "They quit smoking twenty years ago",
]

def _si(value: float, factor: float) -> str:
    """``value * factor`` with the fewest decimals that convert back within 0.5%
    (and back to the same integer for integer fields such as PSI glucose)."""
    for decimals in range(0, 4):
        text = f"{value * factor:.{decimals}f}"
        back = float(text) / factor
        if abs(back - value) <= 0.005 * abs(value) and round(back) == round(value):
            return text
    return f"{value * factor:.3f}"


# (calculator, criterion) -> function(option) -> replacement text
UNIT_CONVERSIONS: Dict[tuple, Callable[[g.Option], str]] = {
    ("meldna", "creatinine"): lambda o: f"{_si(float(o.input_string), 88.4)} µmol/L",
    ("meldna", "bilirubin"): lambda o: f"{_si(float(o.input_string), 17.1)} µmol/L",
    ("sofa", "sofa_creatinine"): lambda o: f"{_si(float(o.input_string), 88.4)} µmol/L",
    ("sofa", "sofa_bilirubin"): lambda o: f"{_si(float(o.input_string), 17.1)} µmol/L",
    ("psiport", "psiport_temperature"): lambda o: f"{float(o.input_string) * 9 / 5 + 32:.1f} °F",
    ("psiport", "psiport_glucose"): lambda o: f"{_si(float(o.input_string), 1 / 18.016)} mmol/L",
    ("hasbled", "hasbled_renal_disease"): lambda o: re.sub(
        r"creatinine is (\d+(?:\.\d+)?)",
        lambda m: f"creatinine is {_si(float(m.group(1)), 88.4)} µmol/L",
        o.input_string,
    ),
}

# criteria whose option text is a bare current lab value
CONFLICT_CRITERIA = {
    "meldna": ["sodium", "bilirubin", "creatinine", "inr"],
    "sofa": ["sofa_platelets", "sofa_bilirubin", "sofa_creatinine"],
    "psiport": ["psiport_sodium", "psiport_bun", "psiport_glucose", "psiport_hematocrit"],
}

ABBREVIATIONS = {
    "history of DVT/PE": "h/o DVT/PE",
    "history of COPD": "h/o COPD",
    "history of inflammatory bowel disease": "h/o IBD",
    "a CHF exacerbation": "a CHF exac",
    "They report drinking nine alcoholic drinks per week": "EtOH 9 drinks/wk",
    "Patient expresses abstinence from alcohol": "No EtOH",
    "They take ibuprofen daily": "Takes ibuprofen qd",
    "The individual states having a previous stroke in their medical history": "PMHx significant for CVA",
    "Patient informs that they have no history of stroke": "No h/o CVA",
    "Last year, they were admitted to a local hospital for a GIB and required 3 units pRBCs": (
        "Admitted last yr for GIB requiring 3u pRBC"
    ),
    "They have active cancer": "Active malignancy",
    "They have no active cancer": "No active malignancy",
    "They have a history of a prior DVT in the femoral vein": "h/o femoral DVT",
    "They have no previously documented DVT": "No prior DVT",
    "They are on three-times-per-week dialysis": "On HD TIW",
    "They are not on dialysis": "Not on HD",
    "The patient reports a myocardial infarction two years ago, followed by coronary stenting": (
        "h/o MI 2y ago s/p PCI"
    ),
    "They deny any history of heart attacks": "Denies h/o MI",
    "The patient reports a long-standing diagnosis of emphysema": "Longstanding emphysema",
    "They deny any history of chronic lung diseases": "Denies chronic lung dz",
    "The patient has end-stage renal disease and undergoes hemodialysis three times a week": "ESRD on HD TIW",
    "They have normal kidney function with no history of renal disease": "Nl renal fxn, no h/o renal dz",
    "The patient has congestive heart failure with an ejection fraction of 35% and takes daily diuretics": (
        "HFrEF (EF 35%) on daily diuretic"
    ),
    "They have no symptoms or diagnosis of heart failure and maintain an active lifestyle": "No h/o CHF",
    "It is observed that the patient's speech is slurred but mostly intelligible": "Mild-mod dysarthria",
    "The patient's speech is clear and easily understood": "Speech clear",
    "No ataxia is detected when the patient performs finger-nose-finger and heel-shin tests": "FNF/HKS w/o ataxia",
}

LONG_NOTE = (
    "CHIEF COMPLAINT: Follow-up.\n\n"
    "HISTORY OF PRESENT ILLNESS:\n{body}\n\n"
    "SOCIAL HISTORY: Lives with spouse. Retired accountant. Walks the dog daily.\n\n"
    "REVIEW OF SYSTEMS: Denies headache, sore throat, rash, or dysuria. Sleeping well.\n\n"
    "MEDICATIONS: See medication reconciliation in the chart.\n\n"
    "PLAN: Discussed with the patient; questions answered.\n\n"
    "{question}"
)


def _case_parts(case: g.GeneratedCase) -> List[str]:
    return _template_parts(case.calculator)[case.template_index]


def render(case: g.GeneratedCase, overrides: Dict[str, str], appended: Dict[str, str] = None,
           order: Optional[Dict[str, str]] = None) -> str:
    """Rebuild the vignette the way the generator does, with some option text replaced.

    ``order`` maps a placeholder to the placeholder whose text goes in its position.
    Substitution mirrors ``generator.build_case`` (str.replace on the raw template),
    so an unperturbed render reproduces the original vignette exactly.
    """
    text = g.templates(case.calculator)[case.template_index]
    by = {o.criterion_slug: o for o in case.options}
    appended = appended or {}
    order = order or {}
    # two passes so that reordering cannot substitute into already-substituted text
    for crit in by:
        text = text.replace(f"{{{crit}}}", f"\x00{crit}\x00")
    for crit, opt in by.items():
        source = order.get(crit, crit)
        value = overrides.get(source, by[source].input_string)
        if source in appended:
            value = f"{value}. {appended[source]}"
        text = text.replace(f"\x00{crit}\x00", value)
    return " ".join(text.split())


def _sentence_slots(case: g.GeneratedCase) -> List[str]:
    """Placeholders that are a whole sentence on their own: '. {slot}.'"""
    parts = _case_parts(case)
    slots = []
    for i in range(1, len(parts), 2):
        before = parts[i - 1].strip()
        after = parts[i + 1] if i + 1 < len(parts) else ""
        if before.endswith((".", "?", "!")) and after.startswith("."):
            slots.append(parts[i])
    return slots


def perturb(case: g.GeneratedCase, kind: str, rng: random.Random) -> Optional[str]:
    by = {o.criterion_slug: o for o in case.options}
    if kind == "distractors":
        slots = _sentence_slots(case)
        if not slots:
            return None
        chosen = rng.sample(slots, k=min(3, len(slots)))
        sentences = rng.sample(DISTRACTORS, k=len(chosen))
        return render(case, {}, appended=dict(zip(chosen, sentences)))
    if kind == "units":
        overrides = {c: f(by[c]) for (calc, c), f in UNIT_CONVERSIONS.items() if calc == case.calculator}
        return render(case, overrides) if overrides else None
    if kind == "conflicting":
        overrides = {}
        for c in CONFLICT_CRITERIA.get(case.calculator, []):
            current = float(by[c].input_string)
            if "sodium" in c:  # keep sodium physiological
                older = current + rng.choice([-8, -5, 5, 8])
            else:
                older = round(current * rng.choice([0.6, 0.75, 1.3, 1.5]), 1)
            older_text = f"{older:g}" if older != int(older) else f"{int(older)}"
            overrides[c] = f"{by[c].input_string} (it was {older_text} three days ago)"
        return render(case, overrides) if overrides else None
    if kind == "abbreviations":
        overrides = {
            c: ABBREVIATIONS[" ".join(o.input_string.split())]
            for c, o in by.items()
            if " ".join(o.input_string.split()) in ABBREVIATIONS
        }
        if not overrides:
            # phrases embedded inside longer option text (e.g. Caprini venous disease)
            for c, o in by.items():
                for long, short in ABBREVIATIONS.items():
                    if long in o.input_string:
                        overrides[c] = o.input_string.replace(long, short)
        return render(case, overrides) if overrides else None
    if kind == "reorder":
        slots = _sentence_slots(case)
        if len(slots) < 3:
            return None
        shuffled = slots[:]
        while shuffled == slots:
            rng.shuffle(shuffled)
        return render(case, {}, order=dict(zip(slots, shuffled)))
    if kind == "long_note":
        text = case.vignette
        m = re.search(r"([^.?!]*\?)\s*$", text)
        if not m:
            return None
        body, question = text[: m.start()].strip(), m.group(1).strip()
        return LONG_NOTE.format(body=body, question=question)
    raise ValueError(kind)


PERTURBATIONS = ["distractors", "units", "conflicting", "abbreviations", "reorder", "long_note"]
