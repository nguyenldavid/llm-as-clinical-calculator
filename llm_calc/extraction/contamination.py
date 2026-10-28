"""
Contamination probe: has a model seen the test notes during training?

T5 (MedCalc-Bench) has been public since mid-2024 and its notes come from
published case reports (PMC), so open models trained after that may have seen
them. T3 paraphrases were written for this study and cannot have been seen,
which makes them the control. T1 and T2 are not probed: they are rendered from
templates, so their continuations are predictable from the prefix whether or
not a model has seen them (and the templates have been public in this
repository since 2024, which is why T1 is never a primary set).

The probe (after Golchin & Surdeanu 2023, "Time travel in LLMs", and the n-gram
checks of Xu et al. 2024): give the model the first half of a note as raw text
(no chat template, no instruction), decode the next ``--tokens`` tokens
greedily, and compare them with the note's true continuation:

* ``rougeL``: LCS-based F1 over word tokens;
* ``longest_ngram``: longest run of consecutive words shared with the true
  continuation, in order;
* ``verbatim``: ``longest_ngram`` >= 13 (the GPT-3 / Llama 2 contamination
  threshold).

Each model gets the same number of exposed (T5) and fresh (T3) notes. If T5
notes are completed verbatim more often than T3 notes, that model's T5 results
are flagged in the paper and the conclusions rest on T2/T3. API models cannot be probed this
way; for them the paper reports the training cut-off and the T5 release date.

    python -m llm_calc.extraction.contamination run --model PATH --out DIR [--sets T5,T3] [--n 200]
    python -m llm_calc.extraction.contamination summarise --dir DIR
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import stats

EXPOSED = {"T5"}  # public before the models' training cut-offs
VERBATIM_NGRAM = 13
MIN_WORDS = 40  # shorter notes leave too little to continue


def words(text: str) -> List[str]:
    return text.split()


def split_note(text: str, fraction: float = 0.5) -> Optional[tuple]:
    """(prefix text, continuation words), cut at a word boundary."""
    w = words(text)
    if len(w) < MIN_WORDS:
        return None
    k = int(len(w) * fraction)
    return " ".join(w[:k]), w[k:]


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: Sequence[str], reference: Sequence[str]) -> float:
    if not candidate or not reference:
        return 0.0
    lcs = _lcs(candidate, reference)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(candidate), lcs / len(reference)
    return 2 * p * r / (p + r)


def longest_ngram(candidate: Sequence[str], reference: Sequence[str]) -> int:
    """Longest run of consecutive words appearing in both, in order."""
    best = 0
    prev = [0] * (len(reference) + 1)
    for x in candidate:
        cur = [0] * (len(reference) + 1)
        for j, y in enumerate(reference):
            if x == y:
                cur[j + 1] = prev[j] + 1
                best = max(best, cur[j + 1])
        prev = cur
    return best


def _norm(ws: Sequence[str]) -> List[str]:
    return [w.lower().strip(".,;:()[]\"'") for w in ws]


def score(completion: str, continuation: Sequence[str]) -> Dict:
    cand = _norm(words(completion))
    ref = _norm(continuation[: max(len(cand), 1) * 2])  # compare against a window the model could have produced
    ngram = longest_ngram(cand, ref)
    return {"rougeL": rouge_l(cand, ref), "longest_ngram": ngram, "verbatim": ngram >= VERBATIM_NGRAM}


def sample(records: List[Dict], n: int, seed: int = 0) -> List[Dict]:
    """Up to ``n`` notes with distinct text, long enough to split."""
    rng = random.Random(seed)
    seen, out = set(), []
    pool = records[:]
    rng.shuffle(pool)
    for r in pool:
        if r["vignette"] in seen or split_note(r["vignette"]) is None:
            continue
        seen.add(r["vignette"])
        out.append(r)
        if len(out) == n:
            break
    return out


def run(model: str, records: List[Dict], tokens: int = 64, **engine_kwargs) -> List[Dict]:
    from vllm import SamplingParams

    from llm_calc.extraction.extractors.llm import load_vllm

    llm = load_vllm(model, **engine_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=tokens)
    splits = [split_note(r["vignette"]) for r in records]
    outputs = llm.generate([prefix for prefix, _ in splits], params)
    rows = []
    for r, (prefix, cont), out in zip(records, splits, outputs):
        text = out.outputs[0].text
        rows.append({"id": r["id"], "test_set": r["test_set"], "exposed": r["test_set"] in EXPOSED,
                     "model": Path(model.rstrip("/")).name, "completion": text[:2000], **score(text, cont)})
    return rows


def summarise(rows: List[Dict]) -> List[Dict]:
    """Per model and set: verbatim rate (Wilson CI) and mean ROUGE-L; plus exposed vs fresh per model."""
    out = []
    groups: Dict[tuple, List[Dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["test_set"]), []).append(r)
        groups.setdefault((r["model"], "exposed" if r["exposed"] else "fresh"), []).append(r)
    for (model, group), rs in sorted(groups.items()):
        k = sum(bool(r["verbatim"]) for r in rs)
        lo, hi = stats.wilson(k, len(rs))
        out.append({"model": model, "group": group, "n": len(rs), "verbatim_rate": k / len(rs),
                    "verbatim_lo": lo, "verbatim_hi": hi,
                    "mean_rougeL": sum(r["rougeL"] for r in rs) / len(rs),
                    "mean_longest_ngram": sum(r["longest_ngram"] for r in rs) / len(rs)})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m llm_calc.extraction.contamination", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("run")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--sets", default="T5,T3")
    s.add_argument("--n", type=int, default=200, help="notes per set")
    s.add_argument("--tokens", type=int, default=64)
    s.add_argument("--data-dir", default=str(ds.DATA_DIR))
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument("--max-model-len", type=int, default=8192)
    s = sub.add_parser("summarise")
    s.add_argument("--dir", required=True)
    args = p.parse_args(argv)
    if args.step == "run":
        records = []
        for name in args.sets.split(","):
            path = Path(args.data_dir) / f"{name}.jsonl"
            if path.exists():
                records += sample(ds.load_set(name, Path(args.data_dir)), args.n)
            else:
                print(f"skipping {name}: not built yet")
        rows = run(args.model, records, args.tokens, tensor_parallel_size=args.tensor_parallel_size,
                   max_model_len=args.max_model_len)
        out = Path(args.out)
        path = ds.write_jsonl(rows, out / f"{Path(args.model.rstrip('/')).name}.jsonl")
        print(f"{len(rows)} completions -> {path}")
    else:
        rows = [r for path in sorted(Path(args.dir).glob("*.jsonl")) for r in ds.read_jsonl(path)]
        summary = summarise(rows)
        with open(Path(args.dir) / "contamination.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]), delimiter="\t")
            w.writeheader()
            w.writerows(summary)
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
