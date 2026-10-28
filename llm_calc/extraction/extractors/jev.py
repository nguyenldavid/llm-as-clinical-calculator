"""
Jev (TypeSafe AI's System One decision model) as an extractor.

Jev answers typed questions about a ``state`` instead of generating text: a
Noul (probability that a statement is true), a Choice (one of up to 255
options) or a Score (a position on up to 10 ordered levels), each with
per-option probabilities and a confidence. That matches this benchmark's
design: Jev makes the narrow judgments, and code (the typed request and
OpenMedCalc) does all arithmetic. TypeSafe's own guidance is the same: keep math
in code, and turn value extraction into a Choice over candidates found by regex.

Field mapping (one request per case; every field is a question, "fan-out"):

  bool (decision schema)        Noul, true when P(yes) > 0.5
  bool, Optional (raw schema)   Choice: present / absent / not documented
  ordered enum                  Score over the levels; the most probable level is
                                taken (TypeSafe advises against interpolating)
  other enum                    Choice over the enum values
  int / float                   regex finds every number in the note; a Choice picks
                                the one that is this field (or "none"); code parses it
                                and converts units (units.normalise)
  SOFA map                      also asks for systolic and diastolic pressure, and
                                computes MAP in code when no MAP is stated

Two per-field numbers are kept in the prediction:

* ``confidence``: Jev's own statistic of how concentrated the answer
  distribution is (1 = all mass on one option, 0 = uniform; for a Noul,
  |2p - 1|). This is what TypeSafe recommends thresholding on, so the
  risk-coverage curves and operating points use it.
* ``probability``: the probability Jev gave the answer it chose (max(p, 1 - p)
  for a Noul). This is the system's stated chance of being right, so the
  calibration metrics (Brier, ECE, reliability diagram) use it.

API: POST https://api.typesafe.ai/v1/systemone with ``Authorization: Bearer
$TYPESAFE_API_KEY``. Only synthetic sets (T1-T4, A409) and T5 go to the API;
real patient notes need a data processing agreement first (TypeSafe offers zero
data retention to enterprise customers).
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from llm_calc.extraction import schemas, units
from llm_calc.extraction.extractors.base import Extractor, prediction

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.14.0"  # pinned; the jev-latest alias moves with releases
NONE = "none"
ZERO = "zero"  # "abstains", "not on pressors": the value is stated, and it is 0
CONTEXT_CHARS = 45

# enums whose values are ordered from least to most severe (Score questions)
ORDINAL_FIELDS = {
    ("cci", "liver_disease"), ("cci", "diabetes"), ("cci", "solid_tumor"), ("caprini", "mobility"),
}
ORDINAL_CALCULATORS = {"nihss", "gad7"}
# integer fields that are really flags (asked as a Noul)
FLAG_INTS = {("psiport", "pleural_effusion")}

NOUL_CRITERIA = ("Answer yes only if the clinical text states this for the patient now or in their history as "
                 "described. Answer no if the text denies it, attributes it to someone else (e.g. a relative), "
                 "or does not mention it.")


def _readable(value: str) -> str:
    return value.replace("_", " ").lower() if value.isupper() else value


def _number_candidates(text: str) -> List[Tuple[str, str, int, int]]:
    """(key, description, start, end) for every number in the text."""
    out = []
    for i, m in enumerate(units.NUMBER.finditer(text)):
        left = text[max(0, m.start() - CONTEXT_CHARS): m.start()].replace("\n", " ")
        right = text[m.end(): m.end() + CONTEXT_CHARS].replace("\n", " ")
        out.append((f"n{i}", f"…{left}[{m.group()}]{right}…", m.start(), m.end()))
    return out[:253]  # Choice allows 255 options, including "zero" and "none"


# a decoder turns the answers back into (field value, confidence)
Decoder = Callable[[Dict], Tuple[object, Optional[float]]]


def build(calculator: str, variant: str, text: str) -> Tuple[Dict, Dict[str, Decoder]]:
    """Questions for one case, and a decoder per schema field."""
    questions: Dict[str, Dict] = {}
    decoders: Dict[str, Decoder] = {}
    candidates = _number_candidates(text)
    span = {key: (start, end) for key, _, start, end in candidates}

    def number_question(qname: str, what: str):
        questions[qname] = {
            "type": "choice",
            "instructions": (f"Which highlighted number in the clinical text is the patient's {what}? "
                             "Pick the current value. If the text says it is zero or absent, pick zero. "
                             "If the text does not give it, pick none."),
            "criteria": {key: desc for key, desc, _, _ in candidates}
            | {ZERO: "The text states it is zero or none (for example abstains, not receiving, no output).",
               NONE: "The text does not state this value."},
        }

    def read_number(answers: Dict, qname: str, field: str) -> Tuple[Optional[float], Optional[float]]:
        a = answers.get(qname, {})
        key = a.get("choice")
        if key == ZERO:
            return 0.0, a.get("confidence")
        if key in (None, NONE) or key not in span:
            return None, a.get("confidence")
        start, end = span[key]
        value = units.number_value(text[start:end])
        return units.normalise(field, value, text[end:]), a.get("confidence")

    for spec in schemas.field_specs(calculator, variant):
        name, desc = spec.name, spec.description
        if spec.kind == "bool" or (calculator, name) in FLAG_INTS:
            if spec.optional:
                questions[name] = {
                    "type": "choice",
                    "instructions": f"For this patient, what does the clinical text say about: {desc}?",
                    "criteria": {"present": "The text states that it applies to the patient.",
                                 "absent": "The text states that it does not apply to the patient.",
                                 "not_documented": "The text does not mention it for the patient."},
                }
                decoders[name] = (lambda a, n=name: (
                    {"present": True, "absent": False}.get(a.get(n, {}).get("choice")), a.get(n, {}).get("confidence")))
            else:
                questions[name] = {"type": "noul", "instructions": f"Does this apply to the patient: {desc}?",
                                   "criteria": NOUL_CRITERIA}
                as_int = (calculator, name) in FLAG_INTS

                def decode_noul(a, n=name, as_int=as_int):
                    p = a.get(n, {}).get("noul")
                    if p is None:
                        return None, None
                    v = p > 0.5
                    return (int(v) if as_int else v), abs(p - 0.5) * 2
                decoders[name] = decode_noul
        elif spec.kind == "enum":
            ordinal = calculator in ORDINAL_CALCULATORS or (calculator, name) in ORDINAL_FIELDS
            if ordinal:
                questions[name] = {"type": "score", "instructions": f"Rate the patient on: {desc}",
                                   "criteria": [_readable(c) for c in spec.choices]}

                def decode_score(a, n=name, choices=spec.choices):
                    ans = a.get(n, {})
                    probs = ans.get("probabilities") or {}
                    if not probs:
                        return None, None
                    level = int(max(probs, key=lambda k: probs[k]))
                    return choices[level], ans.get("confidence")
                decoders[name] = decode_score
            else:
                questions[name] = {"type": "choice", "instructions": f"Which applies to the patient: {desc}?",
                                   "criteria": {c: _readable(c) for c in spec.choices}}
                decoders[name] = lambda a, n=name: (a.get(n, {}).get("choice"), a.get(n, {}).get("confidence"))
        else:
            number_question(name, desc)

            def decode_number(a, n=name, kind=spec.kind, optional=spec.optional):
                value, conf = read_number(a, n, n)
                if value is None and calculator == "sofa" and n == "map":
                    sbp, c1 = read_number(a, "map__sbp", "sbp")
                    dbp, c2 = read_number(a, "map__dbp", "dbp")
                    if sbp is not None and dbp is not None:
                        value, conf = (sbp + 2 * dbp) / 3, min(c for c in (c1, c2) if c is not None)
                if value is None and not optional and n.endswith("_dose"):
                    value = 0.0  # decision-schema convention: a dose not given is 0
                if value is not None and kind == "int":
                    value = int(round(value))
                return value, conf
            decoders[name] = decode_number
            if calculator == "sofa" and name == "map":
                number_question("map__sbp", "systolic blood pressure in mmHg")
                number_question("map__dbp", "diastolic blood pressure in mmHg")
    return questions, decoders


def chosen_probability(answer: Optional[Dict]) -> Optional[float]:
    """Probability of the option or level Jev answered with."""
    if not answer:
        return None
    if answer.get("noul") is not None:
        p = float(answer["noul"])
        return max(p, 1 - p)
    probs = answer.get("probabilities") or {}
    return max(probs.values()) if probs else None


def http_transport(api_key: str, timeout_s: float = 60, retries: int = 5) -> Callable[[Dict], Dict]:
    def send(body: Dict) -> Dict:
        data = json.dumps(body).encode()
        for attempt in range(retries + 1):
            req = urllib.request.Request(API_URL, data=data, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                    retry_after = e.headers.get("retry-after")
                    time.sleep(float(retry_after) if retry_after else 2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
            except urllib.error.URLError:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")
    return send


class JevExtractor(Extractor):
    def __init__(self, model: str = DEFAULT_MODEL, name: Optional[str] = None, api_key_env: str = "TYPESAFE_API_KEY",
                 concurrency: int = 8, transport: Optional[Callable[[Dict], Dict]] = None):
        self.model = model
        self.name = name or f"jev:{model}"
        self.concurrency = concurrency
        if transport is None:
            key = os.environ.get(api_key_env)
            if not key:
                raise SystemExit(f"set {api_key_env} to call the TypeSafe API")
            transport = http_transport(key)
        self.transport = transport

    def _one(self, record: Dict, variant: str) -> Dict:
        calc, text = record["calculator"], record["vignette"]
        questions, decoders = build(calc, variant, text)
        start = time.perf_counter()
        try:
            response = self.transport({"model": self.model, "state": text, "questions": questions})
        except Exception as e:  # noqa: BLE001 - recorded per case
            return prediction(record, self.name, variant, None, error=f"request failed: {e}",
                              latency_s=time.perf_counter() - start)
        answers = response.get("answers", {})
        output, confidence = {}, {}
        for field, decoder in decoders.items():
            output[field], confidence[field] = decoder(answers)
        pred = prediction(record, self.name, variant, output, raw=json.dumps(answers)[:20000],
                          latency_s=time.perf_counter() - start)
        pred["confidence"] = confidence
        pred["probability"] = {field: chosen_probability(answers.get(field)) for field in decoders}
        pred["model_version"] = response.get("model")
        pred["usage"] = response.get("usage")
        return pred

    def extract_batch(self, records, variant):
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(lambda r: self._one(r, variant), records))
