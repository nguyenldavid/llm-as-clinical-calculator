"""
Generate paraphrase candidates for clinician review (GPU job on Killarney).

    python -m llm_calc.extraction.paraphrase options  --model PATH --out options_review.tsv
    python -m llm_calc.extraction.paraphrase vignettes --model PATH --out t3_review.tsv

``options`` rewords every exported option (T2); ``vignettes`` rewrites whole T1
vignettes (T3). Both write a review sheet with an empty ``approved`` column: a
clinician marks each row y/n before ``build-t2`` / ``build-t3`` will use it.

Use a model family that is *not* among the systems under test, so that no
extractor is advantaged by having written its own test text.
"""

import argparse
import csv
import json
from pathlib import Path

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import phrasings

OPTION_PROMPT = """Rewrite the clinical statement below in {n} different ways.

Rules:
- Keep the exact clinical meaning: same findings, same negations, same numbers, same time frames, same side of the body.
- Do not add or remove any clinical fact.
- Vary vocabulary and sentence structure. Make about one third of the rewrites terse clinical shorthand (e.g. "h/o", "s/p", "pt").
- Each rewrite must fit where the original appears in this sentence frame: "{context}"

Statement: "{text}"

Return a JSON object: {{"rewrites": ["...", "..."]}}"""

VIGNETTE_PROMPT = """Rewrite the clinical case below as a clinician might write it in a note.

Rules:
- Keep every clinical fact exactly: findings, negations, numbers and units, time frames, laterality.
- Do not add any new clinical finding, diagnosis, medication or test result.
- You may reorder information and change wording and style; you may drop the request for a score.

Case: "{text}"

Return a JSON object: {{"rewrite": "..."}}"""


def _schema(key: str, array: bool):
    if array:
        return {"type": "object", "properties": {key: {"type": "array", "items": {"type": "string"}}}, "required": [key]}
    return {"type": "object", "properties": {key: {"type": "string"}}, "required": [key]}


def _generate(llm, prompts, schema, temperature: float, max_tokens: int):
    from llm_calc.extraction.extractors.llm import _sampling_params

    params = _sampling_params(schema, max_tokens, temperature=temperature)
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    return [r.outputs[0].text for r in llm.chat(conversations, sampling_params=params, use_tqdm=True)]


def run_options(args, llm, generator_name: str):
    rows = list(csv.DictReader(open(args.options, newline=""), delimiter="\t"))
    prompts = [OPTION_PROMPT.format(n=args.n, context=r["context"], text=r["original"]) for r in rows]
    outputs = _generate(llm, prompts, _schema("rewrites", True), args.temperature, 2048)
    review = []
    for r, text in zip(rows, outputs):
        try:
            rewrites = json.loads(text)["rewrites"]
        except (json.JSONDecodeError, KeyError, TypeError):
            rewrites = []
        for cand in dict.fromkeys(c.strip() for c in rewrites if c and c.strip()):
            if cand.lower() == r["original"].lower():
                continue
            review.append({"key": r["key"], "calculator": r["calculator"], "criterion": r["criterion"],
                           "option_name": r["option_name"], "score_effect": r["score_effect"],
                           "original": r["original"], "candidate": cand, "generator": generator_name,
                           "split": phrasings.assign_split(r["key"], cand), "approved": "", "note": ""})
    return review, phrasings.REVIEW_COLUMNS


def run_vignettes(args, llm, generator_name: str):
    base = ds.load_set(args.base_set, Path(args.data_dir))
    if args.per_calculator:
        seen, picked = {}, []
        for r in base:
            if seen.get(r["calculator"], 0) < args.per_calculator:
                seen[r["calculator"]] = seen.get(r["calculator"], 0) + 1
                picked.append(r)
        base = picked
    prompts = [VIGNETTE_PROMPT.format(text=r["vignette"]) for r in base]
    outputs = _generate(llm, prompts, _schema("rewrite", False), args.temperature, 2048)
    review = []
    for r, text in zip(base, outputs):
        try:
            cand = json.loads(text)["rewrite"].strip()
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            continue
        review.append({"base_id": r["id"], "calculator": r["calculator"], "original": r["vignette"],
                       "candidate": cand, "generator": generator_name, "approved": "", "note": ""})
    return review, phrasings.T3_COLUMNS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["options", "vignettes"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default=str(ds.DATA_DIR))
    p.add_argument("--options", default=str(ds.DATA_DIR / "phrasing" / "options_to_paraphrase.tsv"))
    p.add_argument("--base-set", default="T1")
    p.add_argument("--per-calculator", type=int, default=100)
    p.add_argument("--n", type=int, default=6, help="rewrites per option")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    args = p.parse_args(argv)

    from llm_calc.extraction.extractors.llm import load_vllm

    llm = load_vllm(args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len)
    name = Path(args.model.rstrip("/")).name
    rows, columns = (run_options if args.mode == "options" else run_vignettes)(args, llm, name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} candidates for review -> {out}")


if __name__ == "__main__":
    main()
