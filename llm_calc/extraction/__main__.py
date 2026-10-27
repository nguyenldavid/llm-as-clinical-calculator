"""
Command line for the extraction benchmark.

    python -m llm_calc.extraction build-gold            # Phase 0 round-trip report
    python -m llm_calc.extraction build-sets            # T1, A409, T4
    python -m llm_calc.extraction build-train --per-calculator 2000 --out $SCRATCH/train.jsonl
    python -m llm_calc.extraction export-options        # input sheet for the paraphrase job
    python -m llm_calc.extraction build-t2 --review-sheet reviewed.tsv
    python -m llm_calc.extraction train-encoder --base-model emilyalsentzer/Bio_ClinicalBERT --out DIR
    python -m llm_calc.extraction predict --system vllm --model PATH --sets T1,T4 --out DIR
    python -m llm_calc.extraction evaluate --pred-dir DIR --out DIR

Run ``python -m llm_calc.extraction <command> --help`` for options.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import yaml

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import generator as g


def _sets(names: str, data_dir: Path):
    out = []
    for name in names.split(","):
        out += ds.load_set(name.strip(), data_dir)
    return out


def _train_records(args):
    if getattr(args, "train", None):
        return list(ds.read_jsonl(Path(args.train)))
    held_out = [r["vignette"] for name in ("T1", "A409") for r in ds.load_set(name, Path(args.data_dir))]
    return ds.build_train(getattr(args, "train_per_calculator", 200), held_out)


def cmd_build_gold(args):
    from llm_calc.extraction import gold

    report = {"cases_per_calculator": args.n, "calculators": {}}
    failures = []
    for calc in g.CALCULATOR_ORDER:
        counts = collections.Counter()
        for seed in range(args.n):
            case = g.build_case(calc, seed)
            try:
                ok, computed, original = gold.round_trip(case)
                counts["agree" if ok else "disagree"] += 1
                if not ok:
                    failures.append({"calculator": calc, "seed": seed, "computed": computed, "original": original})
            except Exception as e:  # noqa: BLE001 - reported, not raised, so every calculator is checked
                counts["error"] += 1
                failures.append({"calculator": calc, "seed": seed, "error": repr(e)})
        report["calculators"][calc] = dict(counts)
    report["failures"] = failures[:50]
    report["all_agree"] = not failures
    # YAML, not JSON: *.json is routed through Git LFS in this repository
    out = Path(args.data_dir) / "phase0_round_trip.yaml"
    out.write_text(yaml.safe_dump(report, sort_keys=False))
    print(json.dumps(report["calculators"], indent=2))
    print(f"all agree: {report['all_agree']}  ->  {out}")
    return 0 if report["all_agree"] else 1


def cmd_build_sets(args):
    data_dir = Path(args.data_dir)
    t1 = ds.build_t1(args.t1_per_calculator)
    ds.write_jsonl(t1, data_dir / "T1.jsonl")
    a409 = ds.build_a409()
    ds.write_jsonl(a409["records"], data_dir / "A409.jsonl")
    (data_dir / "A409_match_report.yaml").write_text(yaml.safe_dump(a409["report"], sort_keys=False))
    t4 = ds.build_t4(t1, args.t4_per_calculator)
    ds.write_jsonl(t4, data_dir / "T4.jsonl")
    print(f"T1 {len(t1)}  A409 {len(a409['records'])} {a409['report']}  T4 {len(t4)}")


def cmd_build_train(args):
    from llm_calc.extraction import phrasings

    held_out = [r["vignette"] for name in args.exclude.split(",") for r in ds.load_set(name, Path(args.data_dir))]
    train = ds.build_train(args.per_calculator, held_out)
    if args.review_sheet:
        train += phrasings.train_with_phrasings(train, Path(args.review_sheet))
    ds.write_jsonl(train, Path(args.out))
    print(f"{len(train)} training records -> {args.out}")


def cmd_export_options(args):
    from llm_calc.extraction import phrasings

    out = Path(args.out or Path(args.data_dir) / "phrasing" / "options_to_paraphrase.tsv")
    print(f"{phrasings.export_options(out)} options -> {out}")


def cmd_build_t2(args):
    from llm_calc.extraction import phrasings

    t2 = phrasings.build_t2(Path(args.review_sheet), args.per_calculator)
    ds.write_jsonl(t2, Path(args.data_dir) / "T2.jsonl")
    print(f"T2 {len(t2)}")


def cmd_build_t3(args):
    from llm_calc.extraction import phrasings

    t3 = phrasings.build_t3(Path(args.sheet), ds.load_set("T1", Path(args.data_dir)))
    ds.write_jsonl(t3, Path(args.data_dir) / "T3.jsonl")
    print(f"T3 {len(t3)}")


def cmd_train_encoder(args):
    from llm_calc.extraction.extractors import encoder

    meta = encoder.train(_train_records(args), args.base_model, Path(args.out), epochs=args.epochs,
                         batch_size=args.batch_size, lr=args.lr, max_length=args.max_length, seed=args.seed)
    print(json.dumps({k: v for k, v in meta.items() if k != "heads"}, indent=2))


def _make_extractor(args):
    from llm_calc.extraction.extractors import baselines

    s = args.system
    if s == "oracle":
        return baselines.Oracle()
    if s == "majority":
        return baselines.Majority(_train_records(args))
    if s == "template_inversion":
        return baselines.TemplateInversion(_train_records(args))
    if s == "rules":
        from llm_calc.extraction.extractors.rules import RulesExtractor

        return RulesExtractor()
    if s == "encoder":
        from llm_calc.extraction.extractors.encoder import EncoderExtractor

        return EncoderExtractor(Path(args.model_dir), name=args.name)
    engine = dict(tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization)
    train = _train_records(args) if args.few_shot else []
    if s == "vllm":
        from llm_calc.extraction.extractors.llm import VLLMExtractor

        kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
        return VLLMExtractor(args.model, name=args.name, few_shot=args.few_shot, train_records=train,
                             constrained=not args.unconstrained, chat_template_kwargs=kwargs,
                             trust_remote_code=args.trust_remote_code, **engine)
    if s == "nuextract":
        from llm_calc.extraction.extractors.llm import NuExtractExtractor

        return NuExtractExtractor(args.model, name=args.name, few_shot=args.few_shot, train_records=train,
                                  enable_thinking=args.enable_thinking, constrained=args.constrained, **engine)
    if s == "api":
        from llm_calc.extraction.extractors.llm import OpenAICompatExtractor

        return OpenAICompatExtractor(args.model, args.base_url, api_key_env=args.api_key_env, name=args.name,
                                     few_shot=args.few_shot, train_records=train, concurrency=args.concurrency)
    raise SystemExit(f"unknown system {s}")


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def cmd_predict(args):
    from llm_calc.extraction import schemas

    extractor = _make_extractor(args)
    out_dir = Path(args.out) / _slug(extractor.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_name in args.sets.split(","):
        records = ds.load_set(set_name, Path(args.data_dir))
        for variant in args.variants.split(","):
            todo = [r for r in records if schemas.has_variant(r["calculator"], variant) and variant in r["gold"]]
            if args.limit:
                todo = todo[: args.limit]
            if not todo:
                continue
            preds = extractor.run(todo, variant, batch_size=args.batch_size)
            path = ds.write_jsonl(preds, out_dir / f"{set_name}_{variant}.jsonl")
            failed = sum(p["output"] is None for p in preds)
            print(f"{extractor.name} {set_name} {variant}: {len(preds)} predictions ({failed} without output) -> {path}")


def _write_tsv(rows, path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def cmd_evaluate(args):
    from llm_calc.extraction import evaluate as ev
    from llm_calc.extraction import stats

    data_dir = Path(args.data_dir)
    records = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        if path.stem in {"T1", "T2", "T3", "T4", "A409"}:
            for r in ds.read_jsonl(path):
                records[r["id"]] = r
    field_rows, case_rows = [], []
    for pred_file in sorted(Path(args.pred_dir).glob("*/*.jsonl")):
        preds = list(ds.read_jsonl(pred_file))
        recs = [records[p["id"]] for p in preds if p["id"] in records]
        f, c = ev.grade(recs, preds)
        field_rows += f
        case_rows += c
    if not case_rows:
        raise SystemExit(f"no predictions found under {args.pred_dir}")

    # perturbation is a grouping of its own for T4
    pert = {rid: r["meta"].get("perturbation") for rid, r in records.items()}
    for row in case_rows + field_rows:
        row["perturbation"] = pert.get(row["id"]) or ""

    summary = ev.summarise(field_rows, case_rows)
    for row in summary:
        rows = [c for c in case_rows if c["system"] == row["system"] and c["variant"] == row["variant"]
                and c["test_set"] == row["test_set"] and (row["calculator"] in ("ALL", c["calculator"]))]
        lo, hi = stats.wilson(sum(c["score_correct"] for c in rows), len(rows))
        row["score_acc_ci"] = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else ""
    by_pert = ev.summarise(field_rows, case_rows, by=("system", "variant", "test_set", "perturbation"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_tsv(summary, out / "summary.tsv")
    _write_tsv([r for r in by_pert if r["test_set"] == "T4" and r["calculator"] == "ALL"], out / "t4_by_perturbation.tsv")
    _write_tsv(ev.field_accuracy_by_kind(field_rows), out / "by_field_kind.tsv")
    ds.write_jsonl(case_rows, out / "cases.jsonl")
    ds.write_jsonl(field_rows, out / "fields.jsonl")

    headline = [r for r in summary if r["calculator"] == "ALL"]
    lines = ["| system | variant | set | n | valid | macro field | all fields | score | score 95% CI | band |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in headline:
        lines.append(f"| {r['system']} | {r['variant']} | {r['test_set']} | {r['n_cases']} | {r['valid_rate']:.3f} | "
                     f"{r['macro_field_acc']:.3f} | {r['all_fields_correct']:.3f} | {r['score_acc']:.3f} | "
                     f"{r['score_acc_ci']} | {r['band_acc']:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m llm_calc.extraction", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ds.DATA_DIR))
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("build-gold", help="round-trip check of the gold mappers")
    s.add_argument("--n", type=int, default=1000, help="cases per calculator")
    s.set_defaults(fn=cmd_build_gold)

    s = sub.add_parser("build-sets", help="write T1, A409 and T4")
    s.add_argument("--t1-per-calculator", type=int, default=100)
    s.add_argument("--t4-per-calculator", type=int, default=50)
    s.set_defaults(fn=cmd_build_sets)

    s = sub.add_parser("build-train", help="write a training pool disjoint from the test sets")
    s.add_argument("--per-calculator", type=int, default=2000)
    s.add_argument("--exclude", default="T1,A409", help="sets whose vignettes are removed from training")
    s.add_argument("--review-sheet", help="approved phrasings; adds train-split rephrased copies")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_train)

    s = sub.add_parser("export-options", help="options sheet for the paraphrase job")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_export_options)

    s = sub.add_parser("build-t2", help="T2 from a reviewed phrasing sheet")
    s.add_argument("--review-sheet", required=True)
    s.add_argument("--per-calculator", type=int, default=100)
    s.set_defaults(fn=cmd_build_t2)

    s = sub.add_parser("build-t3", help="T3 from a reviewed whole-vignette paraphrase sheet")
    s.add_argument("--sheet", required=True)
    s.set_defaults(fn=cmd_build_t3)

    s = sub.add_parser("train-encoder", help="fine-tune the multi-head encoder")
    s.add_argument("--base-model", required=True, help="HF id or local path (pre-downloaded on Killarney)")
    s.add_argument("--out", required=True)
    s.add_argument("--train", help="training JSONL from build-train (default: generate a small pool)")
    s.add_argument("--train-per-calculator", type=int, default=200)
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--lr", type=float, default=3e-5)
    s.add_argument("--max-length", type=int, default=512)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train_encoder)

    s = sub.add_parser("predict", help="run one system over test sets")
    s.add_argument("--system", required=True,
                   choices=["oracle", "majority", "template_inversion", "rules", "encoder", "vllm", "nuextract", "api"])
    s.add_argument("--sets", default="T1,A409,T4")
    s.add_argument("--variants", default="D,R")
    s.add_argument("--out", required=True)
    s.add_argument("--name", help="system name in results (default derived from the model)")
    s.add_argument("--limit", type=int, default=0, help="first N records per set (pilot runs)")
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--train", help="training JSONL (majority, template_inversion, few-shot examples)")
    s.add_argument("--train-per-calculator", type=int, default=200)
    s.add_argument("--model", help="model path or id (vllm, nuextract, api)")
    s.add_argument("--model-dir", help="trained encoder directory")
    s.add_argument("--few-shot", type=int, default=0)
    s.add_argument("--unconstrained", action="store_true", help="vllm: disable JSON-schema decoding")
    s.add_argument("--constrained", action="store_true", help="nuextract: add JSON-schema decoding (off by default)")
    s.add_argument("--trust-remote-code", action="store_true")
    s.add_argument("--chat-template-kwargs", help='vllm: JSON, e.g. \'{"enable_thinking": false}\'')
    s.add_argument("--enable-thinking", type=lambda v: v.lower() == "true", default=False,
                   help="nuextract: NuExtract3 reasoning mode (true/false)")
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument("--max-model-len", type=int, default=8192)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    s.add_argument("--base-url", help="api: OpenAI-compatible base URL, e.g. https://openrouter.ai/api/v1")
    s.add_argument("--api-key-env", default="OPENAI_API_KEY")
    s.add_argument("--concurrency", type=int, default=8)
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("evaluate", help="grade every prediction file under --pred-dir")
    s.add_argument("--pred-dir", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_evaluate)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
