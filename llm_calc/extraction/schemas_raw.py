"""
Raw-facts request schemas (schema variant "R").

The OpenMedCalc request models ask the extractor for some clinical decisions:
HAS-BLED's ``hypertension`` means "uncontrolled, systolic > 160 mmHg" and
``alcohol`` means ">= 8 drinks per week". A raw-facts schema asks only for what
the note states and moves the threshold into code, so an extractor is graded on
reading, not on applying cut-offs.

Most OpenMedCalc schemas already take raw values (MELD-Na, SOFA, PSI/PORT and
ARISCAT ages, vitals and labs), so variant R is only needed where a threshold is
folded into a boolean. HAS-BLED is the clearest case and is implemented here.
Fields are Optional: ``None`` means "not documented", which the conversion
treats as the calculator's default (criterion absent).
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from llm_calc.extraction.generator import Option
from llm_calc.extraction.gold import Gold, _Builder, _num


class RawRequestHASBLED(BaseModel):
    age_years: Optional[int] = Field(None, description="Age in years")
    systolic_bp: Optional[int] = Field(None, description="Most recent systolic blood pressure in mmHg")
    creatinine_mg_dl: Optional[float] = Field(None, description="Serum creatinine in mg/dL")
    on_dialysis: Optional[bool] = Field(None, description="Is the patient receiving dialysis?")
    kidney_transplant: Optional[bool] = Field(None, description="Has the patient had a kidney transplant?")
    cirrhosis: Optional[bool] = Field(None, description="Does the patient have cirrhosis?")
    stroke_history: Optional[bool] = Field(None, description="Has the patient ever had a stroke?")
    prior_major_bleeding: Optional[bool] = Field(
        None, description="History of major bleeding or predisposition to bleeding"
    )
    labile_inr: Optional[bool] = Field(
        None, description="Unstable or high INRs (time in therapeutic range < 60%)"
    )
    antiplatelet_or_nsaid_use: Optional[bool] = Field(
        None, description="Takes aspirin, clopidogrel, NSAIDs or other drugs that raise bleeding risk"
    )
    alcoholic_drinks_per_week: Optional[float] = Field(None, description="Alcoholic drinks per week")

    def to_decision(self) -> Dict:
        """The OpenMedCalc HAS-BLED request implied by these facts."""
        renal = bool(self.on_dialysis) or bool(self.kidney_transplant) or (
            self.creatinine_mg_dl is not None and self.creatinine_mg_dl > 2.26
        )
        return {
            "age": self.age_years if self.age_years is not None else 0,
            "hypertension": self.systolic_bp is not None and self.systolic_bp > 160,
            "severe_renal_disease": renal,
            "severe_liver_disease": bool(self.cirrhosis),
            "stroke_history": bool(self.stroke_history),
            "prior_major_bleeding": bool(self.prior_major_bleeding),
            "labile_inr": bool(self.labile_inr),
            "drugs": bool(self.antiplatelet_or_nsaid_use),
            "alcohol": self.alcoholic_drinks_per_week is not None and self.alcoholic_drinks_per_week >= 8,
        }


def hasbled_raw(options: list[Option]) -> Gold:
    b = _Builder(options)
    b.set("age_years", int(_num(b.opt("hasbled_age").input_string)), "hasbled_age")
    # both hypertension options state a clinic blood pressure, e.g. "165/85"
    b.set("systolic_bp", int(_num(b.opt("hasbled_hypertension").input_string.split("pressure is")[-1])), "hasbled_hypertension")
    b.set("creatinine_mg_dl", _num(b.opt("hasbled_renal_disease").input_string.split("creatinine is")[-1]), "hasbled_renal_disease")
    # dialysis and transplant are never mentioned in the HAS-BLED vignettes
    b.set("on_dialysis", None)
    b.set("kidney_transplant", None)
    for field, crit in [
        ("cirrhosis", "hasbled_liver_disease"),
        ("stroke_history", "hasbled_stroke_history"),
        ("prior_major_bleeding", "hasbled_prior_major_bleeding"),
        ("labile_inr", "hasbled_labile_inr"),
        ("antiplatelet_or_nsaid_use", "hasbled_drugs"),
    ]:
        b.flag(field, crit)
    alcohol = b.opt("hasbled_alcohol")
    drinks = {"HASBLED Alcohol": 9.0, "HASBLED No Alcohol": 0.0}[alcohol.name]
    b.set("alcoholic_drinks_per_week", drinks, "hasbled_alcohol")
    return b.gold()


RAW_MODELS = {"hasbled": RawRequestHASBLED}
RAW_MAPPERS = {"hasbled": hasbled_raw}
