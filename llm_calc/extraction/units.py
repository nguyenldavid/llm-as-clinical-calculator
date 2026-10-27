"""
Unit normalisation for extracted numbers.

The request schemas use US units (mg/dL, °C for PSI temperature). Span-based
systems (encoders, rules) read a number and the few characters after it; this
module converts SI or Fahrenheit values when the unit says so. LLM systems are
expected to convert on their own and do not use this.
"""

import re
from typing import Optional

# (field-name fragment, unit pattern, conversion to schema units)
CONVERSIONS = [
    ("creatinine", r"(µ|u|micro)mol", lambda v: v / 88.4),
    ("bilirubin", r"(µ|u|micro)mol", lambda v: v / 17.1),
    ("temperature", r"°\s*F|deg(rees)?\s*F|\bF\b", lambda v: (v - 32) * 5 / 9),
    ("glucose", r"mmol", lambda v: v * 18.016),
]


def normalise(field: str, value: float, following_text: str) -> float:
    unit = following_text[:16]
    for fragment, pattern, convert in CONVERSIONS:
        if fragment in field and re.match(rf"\s*({pattern})", unit, re.IGNORECASE):
            return convert(value)
    return value


WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|\b(?:" + "|".join(WORD_NUMBERS) + r")\b", re.IGNORECASE)


def number_value(token: str) -> float:
    return float(WORD_NUMBERS.get(token.lower(), token))


def parse_number(text: str) -> Optional[float]:
    m = NUMBER.search(text)
    return number_value(m.group()) if m else None
