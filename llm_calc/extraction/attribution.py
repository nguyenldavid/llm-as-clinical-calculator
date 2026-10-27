"""
Error attribution for the released Abacus-409 failures.

With per-field gold labels for each failure, ask: what is the smallest number of
field changes to the gold request that makes OpenMedCalc return the model's
(wrong) answer? A failure reproducible by one changed field is consistent with
a single extraction/interpretation error; one that no change of up to two
fields reproduces points to arithmetic, formula or reporting errors instead.

The alternative values tried for each field are the values that field takes
anywhere in the generated option space, so the search only considers readings a
vignette could support. Attribution is evidence, not proof: a score can be hit
by coincidence. Results are cross-tabulated with the annotators' error classes.

    python -m llm_calc.extraction.attribution --out llm_calc/data/extraction/a409_attribution.md
"""

import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import generator as g
from llm_calc.extraction import gold as gold_mod


def value_space(calculator: str, seeds: range = range(0, 3000)) -> Dict[str, List]:
    seen = defaultdict(dict)
    for seed in seeds:
        req = gold_mod.gold_for(g.build_case(calculator, seed)).request
        for f, v in req.items():
            seen[f][repr(v)] = v
    return {f: list(vals.values()) for f, vals in seen.items()}


def _score(calc: str, req: Dict) -> Optional[int]:
    try:
        return gold_mod.score(calc, req)
    except Exception:  # noqa: BLE001 - out-of-range combinations just do not count
        return None


def attribute(record: Dict, space: Dict[str, List], max_fields: int = 2) -> Dict:
    calc, gold = record["calculator"], record["gold"]["D"]
    try:
        target = float(record["meta"]["llm_answer"])
    except (TypeError, ValueError):
        return {"k": None, "reason": "no numeric answer"}
    if abs(target - round(target)) > 1e-9:
        return {"k": None, "reason": "non-integer answer"}
    target = int(round(target))
    if target == record["gold_score"]:
        return {"k": 0, "fields": []}
    fields = list(gold)
    for k in range(1, max_fields + 1):
        hits = []
        for combo in itertools.combinations(fields, k):
            alternatives = [[v for v in space[f] if v != gold[f]] for f in combo]
            for values in itertools.product(*alternatives):
                req = dict(gold, **dict(zip(combo, values)))
                if _score(calc, req) == target:
                    hits.append(combo)
                    break
        if hits:
            return {"k": k, "fields": sorted(set(itertools.chain(*hits)))}
    return {"k": f">{max_fields}", "fields": []}


def run(max_fields: int = 2) -> Dict:
    records = ds.load_set("A409")
    spaces = {c: value_space(c) for c in {r["calculator"] for r in records}}
    rows = []
    for r in records:
        res = attribute(r, spaces[r["calculator"]], max_fields)
        # control: the same error size in the other direction; if mirrored errors are just
        # as reproducible, single-field attribution says little for that calculator
        control = {"k": None}
        try:
            mirrored = 2 * r["gold_score"] - float(r["meta"]["llm_answer"])
            control = attribute(dict(r, meta=dict(r["meta"], llm_answer=mirrored)), spaces[r["calculator"]], 1)
        except (TypeError, ValueError):
            pass
        rows.append({"id": r["id"], "calculator": r["calculator"], "arm": r["meta"]["arm"],
                     "error_class": r["meta"]["error_class_one"] or "(none)", **res,
                     "mirrored_k1": control.get("k") == 1})
    return {"rows": rows}


def report(rows: List[Dict]) -> str:
    def key(r):
        return "no numeric answer" if r["k"] is None else (f"{r['k']} field" + ("s" if r["k"] != 1 else ""))

    order = ["1 field", "2 fields", ">2 fields", "no numeric answer", "0 fields"]
    total = Counter(key(r) for r in rows)
    lines = ["# Abacus-409: how many field changes reproduce the wrong answer?", "",
             f"{len(rows)} released failures with per-field gold labels (12 rows have no vignette).", "",
             "| fields changed | n | % |", "|---|---|---|"]
    for k in order:
        if total[k]:
            lines.append(f"| {k} | {total[k]} | {100 * total[k] / len(rows):.1f} |")
    numeric = [r for r in rows if r["k"] is not None]
    lines += ["", f"Control: {sum(r['mirrored_k1'] for r in numeric)} of {len(numeric)} numeric failures "
              f"({100 * sum(r['mirrored_k1'] for r in numeric) / len(numeric):.1f}%) would also be reproduced by one "
              "field change if the error had the same size in the opposite direction.", "",
              "| calculator | n | observed 1 field | mirrored 1 field |", "|---|---|---|---|"]
    by_calc = defaultdict(list)
    for r in numeric:
        by_calc[r["calculator"]].append(r)
    for calc, rs in sorted(by_calc.items()):
        obs = sum(r["k"] == 1 for r in rs)
        mir = sum(r["mirrored_k1"] for r in rs)
        lines.append(f"| {calc} | {len(rs)} | {obs} ({100 * obs / len(rs):.0f}%) | {mir} ({100 * mir / len(rs):.0f}%) |")
    lines += ["", "## By annotated error class", "", "| error class | n | 1 field | 2 fields | >2 fields | no numeric answer |",
              "|---|---|---|---|---|---|"]
    by_class = defaultdict(Counter)
    for r in rows:
        by_class[r["error_class"]][key(r)] += 1
    for cls, c in sorted(by_class.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        cells = [f"{c[k]} ({100 * c[k] / n:.0f}%)" for k in ["1 field", "2 fields", ">2 fields", "no numeric answer"]]
        lines.append(f"| {cls} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Fields most often implicated in single-field attributions", "", "| calculator | field | n |", "|---|---|---|"]
    fields = Counter((r["calculator"], f) for r in rows if r["k"] == 1 for f in r["fields"])
    for (calc, f), n in fields.most_common(15):
        lines.append(f"| {calc} | {f} | {n} |")
    lines += ["", "A single-field attribution means some one-field misreading yields exactly the model's answer;",
              "it is consistent with, not proof of, an extraction error. In additive scores several fields can",
              "explain the same shift, and each of them is counted in the field table."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ds.DATA_DIR / "a409_attribution.md"))
    p.add_argument("--max-fields", type=int, default=2)
    args = p.parse_args(argv)
    rows = run(args.max_fields)["rows"]
    text = report(rows)
    Path(args.out).write_text(text)
    ds.write_jsonl(rows, Path(args.out).with_suffix(".jsonl"))
    print(text)


if __name__ == "__main__":
    main()
