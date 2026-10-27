"""
Encoder (BERT-family) extractor: one shared encoder, one head per schema field.

* bool / enum fields: a classification head on the [CLS] vector, with an extra
  "null" class for Optional fields.
* int / float fields: a start/end span head over the tokens, SQuAD-2 style:
  pointing at [CLS] means "not stated". The span text is parsed as a number and
  passed through ``units.normalise``. A required field predicted as not stated
  becomes 0 (doses that are not given).

Span targets are located from the generator: for a training case the source
option of each numeric field is known, so the number is searched only inside
that option's text in the vignette. Records whose text is not a plain
generator rendering (paraphrase-augmented training data) fall back to a unique
match anywhere in the text, and drop the span label when the match is ambiguous.

Train on Killarney with ``jobs/encoder_train.sbatch``; the same code runs on CPU
for smoke tests with a tiny model.
"""

import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm_calc.extraction import generator as g
from llm_calc.extraction import schemas, units
from llm_calc.extraction.extractors.base import Extractor, prediction

NULL = "__null__"
MAX_SPAN_TOKENS = 12
# integer fields that are really flags: classify rather than point at a number
CATEGORICAL_INTS = {("psiport", "pleural_effusion"): ["0", "1"]}


# ------------------------------------------------------------------ label spaces


def head_specs(calculators: List[str], variants=schemas.VARIANTS) -> List[Dict]:
    heads = []
    for calc in calculators:
        for variant in variants:
            if not schemas.has_variant(calc, variant):
                continue
            for f in schemas.field_specs(calc, variant):
                key = f"{calc}__{variant}__{f.name}"
                if (calc, f.name) in CATEGORICAL_INTS:
                    heads.append({"key": key, "calc": calc, "variant": variant, "field": f.name, "kind": "int",
                                  "labels": CATEGORICAL_INTS[(calc, f.name)], "optional": f.optional})
                elif f.kind in ("bool", "enum"):
                    labels = ["false", "true"] if f.kind == "bool" else list(f.choices)
                    if f.optional:
                        labels.append(NULL)
                    heads.append({"key": key, "calc": calc, "variant": variant, "field": f.name,
                                  "kind": f.kind, "labels": labels, "optional": f.optional})
                else:
                    heads.append({"key": key, "calc": calc, "variant": variant, "field": f.name,
                                  "kind": f.kind, "labels": None, "optional": f.optional})
    return heads


def _encode_label(value, head) -> int:
    if value is None:
        return head["labels"].index(NULL)
    if head["kind"] == "bool":
        return head["labels"].index("true" if value else "false")
    if head["kind"] == "int":
        return head["labels"].index(str(int(value)))
    return head["labels"].index(value)


def _decode_label(idx: int, head):
    label = head["labels"][idx]
    if label == NULL:
        return None
    if head["kind"] == "bool":
        return label == "true"
    if head["kind"] == "int":
        return int(label)
    return label


# ------------------------------------------------------------------ span targets


@lru_cache(maxsize=200_000)
def _case(calculator: str, seed: int) -> g.GeneratedCase:
    return g.build_case(calculator, seed)


def slot_spans(record: Dict) -> Optional[Dict[str, Tuple[int, int]]]:
    """Character span of each criterion's text, if the vignette is a plain rendering."""
    case = _case(record["calculator"], record["seed"])
    if case.vignette != record["vignette"]:
        return None
    by = {o.criterion_slug: o for o in case.options}
    template = g.templates(record["calculator"])[case.template_index]
    spans, pos, text = {}, 0, ""
    for i, piece in enumerate(re.split(r"\{(.*?)\}", template)):
        if i % 2:
            value = by[piece].input_string
            spans[piece] = (len(text), len(text) + len(value))
            text += value
        else:
            text += piece
    return spans if text == record["vignette"] else None


def _criterion_of(record: Dict, variant: str, field: str) -> Optional[str]:
    source = record["field_sources"][variant].get(field)
    names = {o.name: o.criterion_slug for o in _case(record["calculator"], record["seed"]).options}
    return names.get(source)


def number_char_span(record: Dict, variant: str, field: str, value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    text = record["vignette"]
    matches = [m for m in units.NUMBER.finditer(text) if abs(units.number_value(m.group()) - float(value)) < 1e-6]
    spans = slot_spans(record)
    crit = _criterion_of(record, variant, field) if spans else None
    if crit and crit in spans:
        lo, hi = spans[crit]
        inside = [m for m in matches if lo <= m.start() and m.end() <= hi]
        if inside:
            return inside[0].span()
    return matches[0].span() if len(matches) == 1 else None


# ------------------------------------------------------------------ model


def _build_model(base_model: str, heads: List[Dict]):
    import torch
    from torch import nn
    from transformers import AutoModel

    class MultiHeadExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_model)
            hidden = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.cls_heads = nn.ModuleDict(
                {h["key"]: nn.Linear(hidden, len(h["labels"])) for h in heads if h["labels"] is not None})
            self.span_heads = nn.ModuleDict(
                {h["key"]: nn.Linear(hidden, 2) for h in heads if h["labels"] is None})

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            states = self.dropout(self.encoder(**kwargs).last_hidden_state)
            return states

    return MultiHeadExtractor()


def _input_text(record: Dict) -> Tuple[str, int]:
    prefix = f"{schemas.CALCULATOR_NAMES[record['calculator']]}: "
    return prefix + record["vignette"], len(prefix)


def _char_to_token(offsets, char_span, shift) -> Optional[Tuple[int, int]]:
    lo, hi = char_span[0] + shift, char_span[1] + shift
    start = end = None
    for t, (a, b) in enumerate(offsets):
        if a == b:  # special tokens
            continue
        if start is None and a <= lo < b:
            start = t
        if a < hi <= b:
            end = t
    return (start, end) if start is not None and end is not None and end >= start else None


def _examples(records, tokenizer, heads_by_calc, max_length):
    examples, dropped = [], 0
    for r in records:
        text, shift = _input_text(r)
        enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        targets = {}
        for h in heads_by_calc[r["calculator"]]:
            gold = r["gold"].get(h["variant"])
            if gold is None:
                continue
            value = gold.get(h["field"])
            if h["labels"] is not None:
                targets[h["key"]] = _encode_label(value, h)
            else:
                cspan = number_char_span(r, h["variant"], h["field"], value)
                if cspan is None and value not in (None, 0, 0.0):
                    dropped += 1  # stated value we could not locate: no span label
                    continue
                tspan = _char_to_token(enc["offset_mapping"], cspan, shift) if cspan else (0, 0)
                if tspan is None:
                    dropped += 1
                    continue
                targets[h["key"]] = tspan
        examples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                         "token_type_ids": enc.get("token_type_ids"), "targets": targets})
    return examples, dropped


def _collate(batch, pad_id):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    types = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        mask[i, :L] = 1
        if b["token_type_ids"] is not None:
            types[i, :L] = torch.tensor(b["token_type_ids"])
    has_types = batch[0]["token_type_ids"] is not None
    return ids, mask, (types if has_types else None)


def train(train_records: List[Dict], base_model: str, out_dir: Path, epochs: int = 4, batch_size: int = 16,
          lr: float = 3e-5, max_length: int = 512, seed: int = 0, device: Optional[str] = None,
          log_every: int = 50) -> Dict:
    import random

    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    calculators = sorted({r["calculator"] for r in train_records})
    heads = head_specs(calculators)
    heads_by_calc = {c: [h for h in heads if h["calc"] == c] for c in calculators}
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = _build_model(base_model, heads).to(device)
    examples, dropped = _examples(train_records, tokenizer, heads_by_calc, max_length)
    calc_of = [r["calculator"] for r in train_records]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(examples) / batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device == "cuda"
    order = list(range(len(examples)))
    step, started = 0, time.time()
    model.train()
    for epoch in range(epochs):
        random.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            batch = [examples[j] for j in idx]
            ids, mask, types = (t.to(device) if t is not None else None for t in _collate(batch, tokenizer.pad_token_id))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                states = model(ids, mask, types)
                cls = states[:, 0]
                losses = []
                for b, j in enumerate(idx):
                    for key, target in examples[j]["targets"].items():
                        if key in model.cls_heads:
                            logits = model.cls_heads[key](cls[b])
                            losses.append(F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([target], device=device)))
                        else:
                            se = model.span_heads[key](states[b]).float()  # (T, 2)
                            se = se.masked_fill(mask[b].unsqueeze(-1) == 0, -1e4)
                            losses.append(F.cross_entropy(se[:, 0].unsqueeze(0), torch.tensor([target[0]], device=device)))
                            losses.append(F.cross_entropy(se[:, 1].unsqueeze(0), torch.tensor([target[1]], device=device)))
                loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} ({time.time() - started:.0f}s)", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    meta = {"base_model": base_model, "heads": heads, "max_length": max_length, "epochs": epochs,
            "batch_size": batch_size, "lr": lr, "seed": seed, "n_train": len(examples),
            "dropped_span_labels": dropped, "calculators": calculators,
            "train_ids_sample": [r["id"] for r in train_records[:5]]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class EncoderExtractor(Extractor):
    def __init__(self, model_dir: Path, name: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.name = name or f"encoder:{Path(self.meta['base_model']).name}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", use_fast=True)
        self.heads = {h["key"]: h for h in self.meta["heads"]}
        self.model = _build_model(self.meta["base_model"], self.meta["heads"])
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()

    def extract_batch(self, records, variant):
        import torch

        start = time.perf_counter()
        texts = [_input_text(r) for r in records]
        enc = self.tokenizer([t for t, _ in texts], truncation=True, max_length=self.meta["max_length"],
                             padding=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in enc.items()}
            states = self.model(inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids"))
            per_case = (time.perf_counter() - start) / max(len(records), 1)
            return [self._decode(b, r, variant, texts[b], states, inputs, offsets, per_case)
                    for b, r in enumerate(records)]

    def _decode(self, b, r, variant, text_and_shift, states, inputs, offsets, per_case):
        import torch

        calc = r["calculator"]
        _, shift = text_and_shift
        request = {}
        for f in schemas.field_specs(calc, variant):
            h = self.heads.get(f"{calc}__{variant}__{f.name}")
            if h is None:
                request[f.name] = None
                continue
            if h["labels"] is not None:
                logits = self.model.cls_heads[h["key"]](states[b, 0])
                request[f.name] = _decode_label(int(logits.argmax()), h)
                continue
            se = self.model.span_heads[h["key"]](states[b]).float()
            valid = inputs["attention_mask"][b].bool()
            s, e = se[:, 0].masked_fill(~valid, -1e4), se[:, 1].masked_fill(~valid, -1e4)
            null_score = float(s[0] + e[0])
            T = int(valid.sum())
            value = None
            if T > 1:
                scores = s[1:T, None] + e[None, 1:T]  # (start, end) over tokens after [CLS]
                pos = torch.arange(T - 1, device=scores.device)
                width = pos[None, :] - pos[:, None]
                scores = scores.masked_fill((width < 0) | (width >= MAX_SPAN_TOKENS), -1e9)
                flat = int(scores.argmax())
                if float(scores.view(-1)[flat]) > null_score:
                    i, j = flat // (T - 1) + 1, flat % (T - 1) + 1
                    a = int(offsets[b, i, 0]) - shift
                    z = int(offsets[b, j, 1]) - shift
                    number = units.parse_number(r["vignette"][max(a, 0):z])
                    if number is not None:
                        value = units.normalise(f.name, number, r["vignette"][z:])
            if value is None and not f.optional:
                value = 0
            if value is not None and f.kind == "int":
                value = int(round(value))
            request[f.name] = value
        return prediction(r, self.name, variant, request, latency_s=per_case)
