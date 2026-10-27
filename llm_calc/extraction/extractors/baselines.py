"""
CPU baselines that bound the benchmark.

* ``oracle`` returns the gold request: checks the evaluation harness (must score 100%).
* ``majority`` predicts each field's most common training value: the floor.
* ``template_inversion`` knows the generator's templates and option phrasings and
  inverts them: it parses the vignette with a regex built from each template,
  maps each slot's text back to its option, and applies the gold mapper. It needs
  no clinical understanding, so its T1 score is the memorisation ceiling, and its
  drop on T2-T4 shows how much of T1 is solvable by string matching alone.
"""

import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

from llm_calc.extraction import generator as g
from llm_calc.extraction import gold as gold_mod
from llm_calc.extraction.extractors.base import Extractor, Timer, prediction
from llm_calc.extraction.schemas_raw import RAW_MAPPERS


class Oracle(Extractor):
    name = "oracle"

    def extract_batch(self, records, variant):
        return [prediction(r, self.name, variant, r["gold"][variant]) for r in records]


class Majority(Extractor):
    name = "majority"

    def __init__(self, train_records: List[Dict]):
        counts = defaultdict(lambda: defaultdict(Counter))
        for r in train_records:
            for variant, req in r["gold"].items():
                for field, value in req.items():
                    counts[(r["calculator"], variant)][field][repr(value)] += 1
                    counts[(r["calculator"], variant)]["__values__"][(field, repr(value))] = value
        self.mode = {}
        for key, per_field in counts.items():
            values = per_field.pop("__values__")
            self.mode[key] = {f: values[(f, c.most_common(1)[0][0])] for f, c in per_field.items()}

    def extract_batch(self, records, variant):
        return [prediction(r, self.name, variant, dict(self.mode[(r["calculator"], variant)])) for r in records]


@lru_cache(maxsize=None)
def _template_parts(calculator: str):
    """Each template as alternating literal text and placeholder names."""
    out = []
    for t in g.templates(calculator):
        parts = re.split(r"\{(.*?)\}", t)  # literals at even indexes, placeholders at odd
        out.append([p if i % 2 else " ".join(p.split()) for i, p in enumerate(parts)])
    return out


@lru_cache(maxsize=None)
def _option_texts(criterion: str) -> List[tuple]:
    """(normalised text, option), including historical wordings."""
    texts = []
    for o in g._options_for(criterion):
        texts.append((" ".join(o.input_string.split()), o))
        for old in g.HISTORICAL_INPUT_STRINGS.get(o.name, []):
            texts.append((" ".join(old.split()), o))
    return texts


def _parse(text: str, parts: List[str], pos: int = 0, i: int = 0) -> Optional[Dict[str, g.Option]]:
    """Match ``text`` against template ``parts``, trying every option text per slot.

    Option texts can contain the same punctuation as the template around them
    (HAS-BLED options span several sentences), so slots are matched against the
    known texts with backtracking rather than with lazy regex groups.
    """
    if i == len(parts):
        return {} if text[pos:].strip() == "" else None
    if i % 2 == 0:
        literal = parts[i]
        rest = text[pos:].lstrip()
        if literal and not rest.startswith(literal.lstrip()):
            return None
        new_pos = len(text) - len(rest) + len(literal.lstrip())
        return _parse(text, parts, new_pos, i + 1)
    rest = text[pos:].lstrip()
    offset = len(text) - len(rest)
    for option_text, option in sorted(_option_texts(parts[i]), key=lambda t: -len(t[0])):
        if rest.startswith(option_text):  # empty option texts (SOFA placeholders) match trivially
            tail = _parse(text, parts, offset + len(option_text), i + 1)
            if tail is not None:
                previous = tail.get(parts[i])
                if previous is None or previous.name == option.name:
                    return {**tail, parts[i]: option}
    return None


class TemplateInversion(Extractor):
    name = "template_inversion"

    def __init__(self, train_records: List[Dict]):
        # fallback for slots that cannot be parsed: the most common training option
        counts = defaultdict(Counter)
        for r in train_records:
            for name in r["options"]:
                counts[r["calculator"]][name] += 1
        self.fallback = {calc: c for calc, c in counts.items()}

    def _options(self, calculator: str, text: str) -> Optional[List[g.Option]]:
        text = " ".join(text.split())
        for parts in _template_parts(calculator):
            parsed = _parse(text, parts)
            if parsed is not None:
                return [parsed[c] for c in g._placeholders(calculator) if c in parsed]
        return None

    def _fallback_options(self, calculator: str) -> List[g.Option]:
        chosen = []
        for crit in g._placeholders(calculator):
            names = {o.name: o for o in g._options_for(crit)}
            best = max(names, key=lambda n: self.fallback[calculator][n])
            chosen.append(names[best])
        return chosen

    def extract_batch(self, records, variant):
        out = []
        for r in records:
            with Timer() as t:
                opts = self._options(r["calculator"], r["vignette"])
                error = None
                if opts is None:
                    error = "no template matched; majority options used"
                    opts = self._fallback_options(r["calculator"])
                if variant == "R":
                    request = RAW_MAPPERS[r["calculator"]](opts).request
                else:
                    request = gold_mod.MAPPERS[r["calculator"]](opts).request
            out.append(prediction(r, self.name, variant, request, error=error, latency_s=t.elapsed))
        return out
