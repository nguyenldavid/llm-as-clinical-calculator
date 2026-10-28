"""
Field-error annotation (plan §10): sampling, inter-annotator agreement, counts.

1. ``sample``: draws a stratified sample of wrong fields from an evaluation
   directory (default 300, balanced over system family x calculator) and writes
   one annotation sheet per annotator. System names are hidden behind a uid so
   annotators are blind to which system made the error; the key is a separate file.
2. Two annotators fill ``code`` (one of ``CODES``) and optionally ``note``.
3. ``agree``: Cohen's kappa between the two sheets, overall and per code, plus
   an adjudication sheet listing every disagreement.
4. ``count``: from the adjudicated sheet (or one annotator's sheet when they
   agree), writes ``error_codes.tsv`` in the format ``figures`` reads.

Sheets are TSV. They contain the source text, so sheets built from T5
(MedCalc-Bench, CC-BY-SA) stay local like T5 itself.

    python -m llm_calc.extraction annotate sample --eval-dir DIR --out sheets/
    python -m llm_calc.extraction annotate agree  --a sheets/annotator_A.tsv --b sheets/annotator_B.tsv --out sheets/
    python -m llm_calc.extraction annotate count  --sheet sheets/adjudicated.tsv --key sheets/key.tsv --out DIR
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from llm_calc.extraction import datasets as ds

CODES = {
    "MISS": "fact present in the text, not extracted (or default given)",
    "HALLUC": "value not supported by the text",
    "NEG": "negation or assertion error (absent read as present or vice versa)",
    "TEMP/UNIT": "wrong time point (old vs current value) or unit/conversion error",
    "THRESH": "fact read correctly, threshold or category applied wrongly",
    "ORD/LAT": "off by one on an ordinal scale, or left/right confused",
    "SCHEMA": "output failed validation (wrong type, out of range, malformed)",
    "GOLD": "the gold label looks wrong; flag for adjudication",
}
SHEET_COLUMNS = ["uid", "test_set", "calculator", "field", "gold", "predicted", "text", "code", "note"]


def _family(system: str) -> str:
    if system == "rules":
        return "Rules"
    for prefix, fam in [("encoder", "Encoder"), ("nuextract", "NuExtract"), ("vllm", "Open LLM"), ("api", "API LLM")]:
        if system.startswith(prefix):
            return fam
    return "Other"


def _records(data_dir: Path) -> Dict[str, Dict]:
    out = {}
    for name in ["T1", "A409", "T2", "T3", "T4", "T5"]:
        path = data_dir / f"{name}.jsonl"
        if path.exists():
            out.update({r["id"]: r for r in ds.read_jsonl(path)})
    return out


def sample(eval_dir: Path, out: Path, n: int = 300, seed: int = 0, data_dir: Path = ds.DATA_DIR,
           annotators=("A", "B")) -> Dict:
    rng = random.Random(seed)
    records = _records(data_dir)
    strata = defaultdict(list)
    for row in ds.read_jsonl(eval_dir / "fields.jsonl"):
        # reference baselines (oracle, majority, template inversion) are not systems under comparison
        if row["correct"] or row["variant"] != "D" or row["id"] not in records or _family(row["system"]) == "Other":
            continue
        if "gold" not in row:
            raise ValueError("fields.jsonl lacks gold/pred values; use the output of `evaluate`, not mock data")
        strata[(_family(row["system"]), row["calculator"])].append(row)
    for rows in strata.values():
        rng.shuffle(rows)
    # round-robin over strata so small families and calculators are represented
    picked, keys = [], sorted(strata)
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    rng.shuffle(picked)

    out.mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    for row in picked:
        uid = hashlib.sha1(f"{row['system']}|{row['id']}|{row['field']}".encode()).hexdigest()[:10]
        rec = records[row["id"]]
        sheet.append({"uid": uid, "test_set": row["test_set"], "calculator": row["calculator"], "field": row["field"],
                      "gold": json.dumps(row["gold"]), "predicted": json.dumps(row["pred"]),
                      "text": " ".join(rec["vignette"].split()), "code": "", "note": ""})
        key.append({"uid": uid, "system": row["system"], "family": _family(row["system"]), "id": row["id"]})
    for a in annotators:
        _write(out / f"annotator_{a}.tsv", sheet, SHEET_COLUMNS)
    _write(out / "key.tsv", key, ["uid", "system", "family", "id"])
    (out / "CODES.md").write_text("# Error codes\n\n" + "\n".join(f"- **{c}**: {d}" for c, d in CODES.items()) + "\n")
    return {"sampled": len(sheet), "strata": len(keys), "annotators": list(annotators)}


def _write(path: Path, rows: List[Dict], columns: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> Dict[str, Dict]:
    with open(path, newline="") as f:
        return {r["uid"]: r for r in csv.DictReader(f, delimiter="\t")}


def cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agree(path_a: Path, path_b: Path, out: Path) -> Dict:
    a, b = _read(path_a), _read(path_b)
    uids = [u for u in a if u in b and a[u]["code"].strip() and b[u]["code"].strip()]
    unknown = {r["code"].strip() for s in (a, b) for r in s.values() if r["code"].strip()} - set(CODES)
    if unknown:
        raise ValueError(f"unknown codes in sheets: {sorted(unknown)}")
    ca = [a[u]["code"].strip() for u in uids]
    cb = [b[u]["code"].strip() for u in uids]
    per_code = {}
    for code in CODES:
        per_code[code] = cohen_kappa([c == code for c in ca], [c == code for c in cb])
    disagreements = [dict(a[u], code_A=a[u]["code"], code_B=b[u]["code"], code="")
                     for u in uids if a[u]["code"].strip() != b[u]["code"].strip()]
    agreed = [dict(a[u], code=a[u]["code"].strip()) for u in uids if a[u]["code"].strip() == b[u]["code"].strip()]
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "adjudicate.tsv", disagreements, SHEET_COLUMNS[:-2] + ["code_A", "code_B", "code", "note"])
    _write(out / "agreed.tsv", agreed, SHEET_COLUMNS)
    report = {"double_coded": len(uids), "raw_agreement": sum(x == y for x, y in zip(ca, cb)) / max(len(uids), 1),
              "kappa": cohen_kappa(ca, cb), "kappa_by_code": per_code, "to_adjudicate": len(disagreements)}
    (out / "agreement.json").write_text(json.dumps(report, indent=2))
    return report


def count(sheets: List[Path], key_path: Path, out: Path) -> Dict:
    key = _read(key_path)
    counts = Counter()
    for path in sheets:
        for uid, row in _read(path).items():
            code = row["code"].strip()
            if code and code != "GOLD" and uid in key:
                counts[(key[uid]["family"], code)] += 1
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "error_codes.tsv", [{"family": f, "code": c, "n": n} for (f, c), n in sorted(counts.items())],
           ["family", "code", "n"])
    return {"coded_errors": sum(counts.values())}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m llm_calc.extraction annotate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--eval-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("agree")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("count")
    s.add_argument("--sheet", action="append", required=True, help="agreed.tsv and the adjudicated sheet")
    s.add_argument("--key", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.step == "sample":
        print(sample(Path(args.eval_dir), Path(args.out), args.n, args.seed))
    elif args.step == "agree":
        print(json.dumps(agree(Path(args.a), Path(args.b), Path(args.out)), indent=2))
    else:
        print(count([Path(x) for x in args.sheet], Path(args.key), Path(args.out)))


if __name__ == "__main__":
    main()
