"""
Schema access shared by every extractor: the request model for a calculator and
schema variant, a flat description of its fields, the JSON schema used for
constrained decoding, and the instruction prompt.

Variant "D" is the OpenMedCalc request model unchanged; variant "R" is the
raw-facts model where one exists (see schemas_raw.py).
"""

import enum
import json
import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from pydantic import BaseModel

from llm_calc.extraction import omc
from llm_calc.extraction.schemas_raw import RAW_MODELS

VARIANTS = ("D", "R")


def has_variant(calculator: str, variant: str) -> bool:
    return variant == "D" or (variant == "R" and calculator in RAW_MODELS)


def model_for(calculator: str, variant: str = "D") -> Type[BaseModel]:
    if variant == "D":
        return omc.request_model(calculator)
    if variant == "R" and calculator in RAW_MODELS:
        return RAW_MODELS[calculator]
    raise KeyError(f"no schema variant {variant!r} for {calculator}")


@dataclass
class FieldSpec:
    name: str
    kind: str  # bool | int | float | enum
    choices: Optional[List[str]]  # enum values (JSON form)
    description: str
    optional: bool


def _unwrap(annotation):
    """(inner type, is_optional) for Optional[X] / X | None."""
    args = typing.get_args(annotation)
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        return inner[0], True
    return annotation, False


def field_specs(calculator: str, variant: str = "D") -> List[FieldSpec]:
    specs = []
    for name, info in model_for(calculator, variant).model_fields.items():
        tp, optional = _unwrap(info.annotation)
        if isinstance(tp, type) and issubclass(tp, enum.Enum):
            kind, choices = "enum", [m.value for m in tp]
        elif tp is bool:
            kind, choices = "bool", None
        elif tp is int:
            kind, choices = "int", None
        elif tp is float:
            kind, choices = "float", None
        else:
            raise TypeError(f"{calculator}.{name}: unsupported field type {tp}")
        description = info.description or info.title or name
        if info.title and info.description and info.title not in info.description:
            description = f"{info.title}: {info.description}"
        specs.append(FieldSpec(name, kind, choices, description, optional or not info.is_required()))
    return specs


def json_schema(calculator: str, variant: str = "D") -> Dict:
    return model_for(calculator, variant).model_json_schema()


CALCULATOR_NAMES = {
    "caprini": "Caprini Score for Venous Thromboembolism",
    "wellsdvt": "Wells' Criteria for DVT",
    "meldna": "MELD-Na",
    "sofa": "Sequential Organ Failure Assessment (SOFA) Score",
    "cci": "Charlson Comorbidity Index",
    "ariscat": "ARISCAT Score for Postoperative Pulmonary Complications",
    "psiport": "PSI/PORT Score (Pneumonia Severity Index)",
    "gad7": "GAD-7 (General Anxiety Disorder-7)",
    "hasbled": "HAS-BLED Score for Major Bleeding Risk",
    "nihss": "NIH Stroke Scale/Score (NIHSS)",
}

SYSTEM_PROMPT = (
    "You extract structured clinical data from text. Read the clinical text and fill in "
    "every field of the JSON object described below, using only information stated in "
    "the text. Do not calculate the score. Return only the JSON object."
)


def field_guide(calculator: str, variant: str = "D") -> str:
    lines = []
    for f in field_specs(calculator, variant):
        kind = f"one of {json.dumps(f.choices)}" if f.kind == "enum" else f.kind
        if f.optional:
            kind += " or null if not documented"
        lines.append(f"- {f.name} ({kind}): {f.description}")
    return "\n".join(lines)


def user_prompt(calculator: str, vignette: str, variant: str = "D") -> str:
    # variant D fields are all required, so the convention for absent criteria is stated
    absent = (
        "Every field is required: answer false for a yes/no criterion the text does not mention, "
        "and 0 for a medication dose that is not given.\n\n"
        if variant == "D"
        else ""
    )
    return (
        f"Calculator: {CALCULATOR_NAMES[calculator]}\n\n"
        f"Fields:\n{field_guide(calculator, variant)}\n\n{absent}"
        f"Clinical text:\n{vignette}\n\nJSON:"
    )
