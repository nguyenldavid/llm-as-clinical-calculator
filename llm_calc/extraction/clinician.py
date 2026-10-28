"""
Clinician reference arm: people fill the same typed request as the systems.

Without a human reference, "90% field accuracy" has no anchor: some notes are
ambiguous and some gold labels are wrong. Two clinicians, blind to the gold
labels and to each other, extract the fields for a stratified sample of cases.
Their sheets are imported as two ordinary systems (``clinician:A``,
``clinician:B``) and graded by the same evaluator, which gives:

* clinician accuracy against gold, with the same CIs as every system;
* each system family against the clinicians on the same cases (paired);
* clinician-clinician agreement per field (Cohen's kappa), an upper bound on
  how well the task is defined;
* time per case (self-recorded), against system latency.

Steps:

1. ``sample`` draws ``--per-calculator`` cases per calculator from each set
   (default T2, T3, T5; the primary sets plus real notes) and writes one sheet
   per calculator per reader. Each row is a case: id, note, one column per
   field (allowed values in the header row below it), and ``seconds``.
   The order of cases is shuffled differently for each reader.
2. Clinicians fill the sheets. Blank means "not documented", which the
   decision schema scores as the calculator default (absent / 0).
3. ``import`` turns each reader's sheets into a prediction file that
   ``evaluate`` reads like any other system.

Sheets contain note text, so sheets with T5 rows stay local like T5 itself.

    python -m llm_calc.extraction clinician sample --out sheets/ [--sets T2,T3,T5] [--per-calculator 10]
    python -m llm_calc.extraction clinician import --sheets sheets/reader_A --reader A --out preds/
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import schemas
from llm_calc.extraction.annotation import cohen_kappa

READERS = ("A", "B")
TRUE_WORDS = {"yes", "y", "true", "t", "1", "present"}
FALSE_WORDS = {"no", "n", "false", "f", "0", "absent"}


def sample_cases(records: List[Dict], per_calculator: int, seed: int = 0) -> List[Dict]:
    """``per_calculator`` cases per (set, calculator), at most one per base case."""
    rng = random.Random(seed)
    by_stratum: Dict[tuple, List[Dict]] = {}
    for r in records:
        by_stratum.setdefault((r["test_set"], r["calculator"]), []).append(r)
    picked, used_clusters = [], set()
    for key in sorted(by_stratum):
        pool = by_stratum[key][:]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            cluster = (r.get("meta") or {}).get("base_id", r["id"])
            if cluster in used_clusters:
                continue
            used_clusters.add(cluster)
            picked.append(r)
            taken += 1
            if taken == per_calculator:
                break
    return picked


def _allowed(spec: schemas.FieldSpec) -> str:
    if spec.kind == "bool":
        return "yes / no (blank = not documented)"
    if spec.kind == "enum":
        return " | ".join(spec.choices)
    return f"{spec.kind} (as written, in the schema's unit; blank = not documented)"


def write_sheets(cases: List[Dict], out: Path, readers=READERS, seed: int = 0) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for reader in readers:
        rng = random.Random(f"{seed}:{reader}")
        rdir = out / f"reader_{reader}"
        rdir.mkdir(exist_ok=True)
        for calc in sorted({c["calculator"] for c in cases}):
            rows = [c for c in cases if c["calculator"] == calc]
            rng.shuffle(rows)
            specs = schemas.field_specs(calc, "D")
            with open(rdir / f"{calc}.tsv", "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["id", "note"] + [s.name for s in specs] + ["seconds"])
                w.writerow(["(allowed)", "(read the note; do not edit)"] + [_allowed(s) for s in specs]
                           + ["time spent on this case"])
                w.writerow(["(meaning)", ""] + [s.description for s in specs] + [""])
                for c in rows:
                    w.writerow([c["id"], c["vignette"].replace("\t", " ").replace("\n", " ")] + [""] * len(specs)
                               + [""])
                    counts[reader] += 1
    (out / "instructions.md").write_text(INSTRUCTIONS)
    return {"cases": len(cases), "rows_per_reader": dict(counts)}


INSTRUCTIONS = """# Extraction sheets: instructions

You will read short clinical notes and fill one row per note. Work alone; do not
discuss notes with the other reader until both sheets are returned.

* Each sheet is one calculator. The first three rows give the field names, the
  allowed values and what each field means.
* Fill a field only from what the note says. Leave it blank when the note does
  not mention it (blank is scored as "absent" / 0, as the calculator does).
* Fields with a list of allowed values (categories) must always be filled with
  one of them, including the "none" category when the note mentions nothing.
* Do not compute the score. Do not look anything up beyond the note.
* Numbers: write the value as given, in the unit the field asks for. If the note
  gives another unit, convert it.
* Record the seconds you spent on each note in `seconds` (a stopwatch is fine).
"""


def _parse(spec: schemas.FieldSpec, text: str):
    t = text.strip()
    if not t:
        if spec.kind == "bool":
            return False
        if spec.kind in ("int", "float"):
            return None if spec.optional else 0
        return None
    if spec.kind == "bool":
        low = t.lower()
        if low in TRUE_WORDS:
            return True
        if low in FALSE_WORDS:
            return False
        raise ValueError(f"{spec.name}: {t!r} is not yes/no")
    if spec.kind == "int":
        return int(round(float(t)))
    if spec.kind == "float":
        return float(t)
    return t


def read_sheets(sheet_dir: Path, reader: str) -> List[Dict]:
    """Predictions in the evaluator's format, one per filled row."""
    preds = []
    for path in sorted(Path(sheet_dir).glob("*.tsv")):
        calc = path.stem
        specs = {s.name: s for s in schemas.field_specs(calc, "D")}
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            if row["id"].startswith("("):
                continue
            output, errors = {}, []
            for name, spec in specs.items():
                try:
                    value = _parse(spec, row.get(name) or "")
                except ValueError as e:
                    errors.append(str(e))
                    continue
                if value is not None:
                    output[name] = value
            seconds = row.get("seconds", "").strip()
            preds.append({"id": row["id"], "system": f"clinician:{reader}", "variant": "D",
                          "output": None if errors else output, "raw": json.dumps(row)[:5000],
                          "error": "; ".join(errors) or None,
                          "latency_s": float(seconds) if seconds else None})
    return preds


def agreement(preds_a: List[Dict], preds_b: List[Dict], records: Dict[str, Dict]) -> List[Dict]:
    """Per-field Cohen's kappa and raw agreement between two readers."""
    b = {p["id"]: p for p in preds_b}
    per_field: Dict[tuple, List] = {}
    for pa in preds_a:
        pb = b.get(pa["id"])
        if pb is None or pa["output"] is None or pb["output"] is None:
            continue
        calc = records[pa["id"]]["calculator"]
        for spec in schemas.field_specs(calc, "D"):
            va, vb = pa["output"].get(spec.name), pb["output"].get(spec.name)
            per_field.setdefault((calc, spec.name), []).append((json.dumps(va), json.dumps(vb)))
    out = []
    for (calc, field), pairs in sorted(per_field.items()):
        xa, xb = [p[0] for p in pairs], [p[1] for p in pairs]
        out.append({"calculator": calc, "field": field, "n": len(pairs),
                    "agreement": sum(x == y for x, y in pairs) / len(pairs), "kappa": cohen_kappa(xa, xb)})
    return out


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(prog="python -m llm_calc.extraction clinician", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ds.DATA_DIR))
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--sets", default="T2,T3,T5")
    s.add_argument("--per-calculator", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("import")
    s.add_argument("--sheets", required=True, help="one reader's directory of sheets")
    s.add_argument("--reader", required=True)
    s.add_argument("--out", required=True, help="prediction directory read by `evaluate`")
    s = sub.add_parser("agree")
    s.add_argument("--a", required=True, help="reader A prediction file")
    s.add_argument("--b", required=True, help="reader B prediction file")
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir)
    if args.step == "sample":
        records = []
        for name in args.sets.split(","):
            path = data_dir / f"{name}.jsonl"
            if not path.exists():
                print(f"skipping {name}: {path} does not exist yet")
                continue
            records += ds.load_set(name, data_dir)
        cases = sample_cases(records, args.per_calculator, args.seed)
        print(write_sheets(cases, Path(args.out), seed=args.seed))
    elif args.step == "import":
        preds = read_sheets(Path(args.sheets), args.reader)
        by_set: Dict[str, List[Dict]] = {}
        for pr in preds:
            by_set.setdefault(pr["id"].split("-")[0], []).append(pr)
        out = Path(args.out) / f"clinician_{args.reader}"
        for set_name, rows in by_set.items():
            ds.write_jsonl(rows, out / f"{set_name}_D.jsonl")
        print({k: len(v) for k, v in by_set.items()})
    else:
        records = {}
        for path in data_dir.glob("*.jsonl"):
            for r in ds.read_jsonl(path):
                records[r["id"]] = r
        rows = agreement(list(ds.read_jsonl(Path(args.a))), list(ds.read_jsonl(Path(args.b))), records)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "clinician_agreement.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["calculator", "field", "n", "agreement", "kappa"], delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} fields -> {out / 'clinician_agreement.tsv'}")


if __name__ == "__main__":
    main()
