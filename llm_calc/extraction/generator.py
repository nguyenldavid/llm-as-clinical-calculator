"""
Vignette generation that reproduces ``llm_calc.lib.vignette`` exactly, without
the rest of the experiment stack (config, LLM clients, DuckDB).

The original generator:

1. picks one template for the calculator with ``templates.sample(n=1, random_state=seed)``;
2. sorts every placeholder that appears in any of that calculator's templates;
3. for each placeholder ``c`` (in sorted order) samples one criterion option with
   ``options[options.criterion_slug == c].sample(n=1, random_state=seed + k)``;
4. substitutes ``input_string`` into the template.

``DataFrame.sample(n=1, random_state=s)`` without weights is
``RandomState(s).permutation(len(df))[0]``; :func:`_pick` uses that directly
so that matching thousands of seeds against released vignettes stays fast.
``tests`` check the shortcut against pandas.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from llm_calc.extraction import omc

BUILD_DIR = Path(__file__).resolve().parents[1] / "data" / "build_database"
XLSX_PATH = BUILD_DIR / "data.xlsx"
TEMPLATE_DIR = BUILD_DIR / "vignette_templates"

# order of the calculator sheet, which is the order the original experiment
# assigned consecutive case ids in (see vignette.gen_all_cases)
CALCULATOR_ORDER = [
    "psiport",
    "ariscat",
    "cci",
    "caprini",
    "gad7",
    "sofa",
    "meldna",
    "hasbled",
    "wellsdvt",
    "nihss",
]


# Option wording that the released Abacus-409 vignettes use but data.xlsx no
# longer does (the spreadsheet was edited after the experiment). Used only to
# match released vignettes back to their seeds; the option slot, and therefore
# its points and gold value, is unchanged.
HISTORICAL_INPUT_STRINGS: Dict[str, List[str]] = {
    "CCI no Liver Disease": ["The patient's liver function tests are normal"],
    "CCI Mild Liver Disease": [
        "They endorse a diagnosis of fatty liver disease and are on a weight management program"
    ],
    "CCI Moderate or Severe Liver Disease": [
        "The patient reports cirrhosis due to chronic alcohol abuse and is awaiting liver transplant evaluation"
    ],
}


@dataclass
class Option:
    id: int
    name: str
    input_string: str
    criterion_slug: str
    calculator_slug: str
    score_effect: int


@dataclass
class GeneratedCase:
    calculator: str
    seed: int
    template_index: int
    vignette: str
    options: List[Option]
    original_score: str  # ground truth exactly as calculate_output_for_case computes it
    extra: Dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def criterion_options() -> pd.DataFrame:
    df = pd.read_excel(XLSX_PATH, sheet_name="criterion_option")
    df["input_string"] = df["input_string"].astype(str)
    return df


@lru_cache(maxsize=1)
def calculator_table() -> pd.DataFrame:
    return pd.read_excel(XLSX_PATH, sheet_name="calculator")


@lru_cache(maxsize=None)
def templates(calculator: str) -> List[str]:
    rows = yaml.safe_load((TEMPLATE_DIR / f"{calculator}.yaml").read_bytes())
    return [row["template"] for row in rows]


@lru_cache(maxsize=None)
def _placeholders(calculator: str) -> List[str]:
    found = set()
    for t in templates(calculator):
        found.update(re.findall(r"\{(.*?)}", t))
    return sorted(found)


@lru_cache(maxsize=None)
def _options_for(criterion: str) -> List[Option]:
    df = criterion_options()
    rows = df[df.criterion_slug == criterion]
    return [Option(**{k: row[k] for k in Option.__dataclass_fields__}) for _, row in rows.iterrows()]


def _pick(n: int, seed: int) -> int:
    return int(np.random.RandomState(seed).permutation(n)[0])


def build_case(calculator: str, seed: int) -> GeneratedCase:
    """Rebuild the case the original generator produces for ``seed``."""
    ts = templates(calculator)
    t_index = _pick(len(ts), seed)
    text = ts[t_index]
    chosen: List[Option] = []
    for k, criterion in enumerate(_placeholders(calculator)):
        pool = _options_for(criterion)
        opt = pool[_pick(len(pool), seed + k)]
        chosen.append(opt)
        text = text.replace(f"{{{criterion}}}", opt.input_string)
    return GeneratedCase(
        calculator=calculator,
        seed=seed,
        template_index=t_index,
        vignette=text,
        options=chosen,
        original_score=original_ground_truth(calculator, chosen),
    )


def original_ground_truth(calculator: str, options: List[Option]) -> str:
    """Mirror of ``vignette.calculate_output_for_case``."""
    if calculator != "meldna":
        return str(int(sum(o.score_effect for o in options)))
    by = {o.criterion_slug: o for o in options}
    request = omc.request_model("meldna")(
        creatinine=float(by["creatinine"].input_string),
        inr=float(by["inr"].input_string),
        bilirubin=float(by["bilirubin"].input_string),
        sodium=float(by["sodium"].input_string),
        is_on_dialysis=by["is_on_dialysis"].name == "Is_On_Dialysis_True",
    )
    return str(omc.calculate("meldna", request))


def original_case_seeds(cases_per_calculator: int, starting_case_id: int) -> Dict[str, List[int]]:
    """Seeds the original ``gen_all_cases`` assigns: consecutive ids per calculator."""
    seeds, case_id = {}, starting_case_id
    for calc in CALCULATOR_ORDER:
        seeds[calc] = list(range(case_id, case_id + cases_per_calculator))
        case_id += cases_per_calculator
    return seeds


def _normalise(text: str) -> str:
    return " ".join(text.split())


def historical_renderings(case: GeneratedCase) -> List[str]:
    """The case's vignette with every combination of historical option wording."""
    renderings = [case.vignette]
    for opt in case.options:
        for old in HISTORICAL_INPUT_STRINGS.get(opt.name, []):
            renderings += [r.replace(opt.input_string, old) for r in renderings]
    return renderings


def seed_index(calculators: List[str], search: range) -> Dict[str, List[GeneratedCase]]:
    """Normalised vignette text -> generated cases that render to it."""
    index: Dict[str, List[GeneratedCase]] = {}
    for calc in calculators:
        for seed in search:
            case = build_case(calc, seed)
            for text in historical_renderings(case):
                index.setdefault(_normalise(text), []).append(case)
    return index
