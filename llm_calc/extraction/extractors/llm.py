"""
LLM extractors.

* ``VLLMExtractor``   offline vLLM (Killarney GPUs): any chat model, JSON-schema
                      constrained decoding, zero- or few-shot.
* ``NuExtractExtractor`` NuExtract 2.0 / NuExtract3 through vLLM, using the
                      model's own JSON-template format.
* ``OpenAICompatExtractor`` any OpenAI-compatible endpoint (OpenAI, OpenRouter,
                      a ``vllm serve`` instance) with ``response_format=json_schema``.
                      Never point this at real patient notes unless the data use
                      agreement allows that provider.

vLLM is imported lazily so the rest of the package works without a GPU.
"""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from llm_calc.extraction import schemas
from llm_calc.extraction.extractors.base import Extractor, parse_json_object, prediction


def few_shot_examples(train_records: List[Dict], calculator: str, variant: str, k: int) -> List[Dict]:
    """First ``k`` training records for the calculator (deterministic)."""
    return [r for r in train_records if r["calculator"] == calculator][:k]


def chat_messages(record: Dict, variant: str, examples: List[Dict]) -> List[Dict]:
    calc = record["calculator"]
    messages = [{"role": "system", "content": schemas.SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": schemas.user_prompt(calc, ex["vignette"], variant)})
        messages.append({"role": "assistant", "content": json.dumps(ex["gold"][variant])})
    messages.append({"role": "user", "content": schemas.user_prompt(calc, record["vignette"], variant)})
    return messages


def _sampling_params(json_schema: Optional[Dict], max_tokens: int, temperature: float = 0.0):
    from vllm import SamplingParams

    kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    if json_schema is None:
        return SamplingParams(**kwargs)
    try:  # vLLM >= 0.10.2
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(structured_outputs=StructuredOutputsParams(json=json_schema), **kwargs)
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(guided_decoding=GuidedDecodingParams(json=json_schema), **kwargs)


def load_vllm(model: str, tensor_parallel_size: int = 1, max_model_len: int = 8192,
              gpu_memory_utilization: float = 0.90, trust_remote_code: bool = False, **kwargs):
    from vllm import LLM

    return LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
               gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=trust_remote_code, **kwargs)


class VLLMExtractor(Extractor):
    def __init__(self, model: str, name: Optional[str] = None, few_shot: int = 0,
                 train_records: Optional[List[Dict]] = None, constrained: bool = True,
                 max_tokens: int = 1024, chat_template_kwargs: Optional[Dict] = None, **engine_kwargs):
        self.name = name or f"vllm:{os.path.basename(model.rstrip('/'))}" + (f":{few_shot}shot" if few_shot else "")
        self.llm = load_vllm(model, **engine_kwargs)
        self.few_shot = few_shot
        self.train = train_records or []
        self.constrained = constrained
        self.max_tokens = max_tokens
        self.chat_template_kwargs = chat_template_kwargs or {}

    def extract_batch(self, records, variant):
        out = []
        by_calc: Dict[str, List[Dict]] = {}
        for r in records:
            by_calc.setdefault(r["calculator"], []).append(r)
        for calc, group in by_calc.items():
            schema = schemas.json_schema(calc, variant) if self.constrained else None
            params = _sampling_params(schema, self.max_tokens)
            examples = few_shot_examples(self.train, calc, variant, self.few_shot)
            conversations = [chat_messages(r, variant, examples) for r in group]
            start = time.perf_counter()
            results = self.llm.chat(conversations, sampling_params=params, use_tqdm=False,
                                    chat_template_kwargs=self.chat_template_kwargs or None)
            per_case = (time.perf_counter() - start) / max(len(group), 1)
            for r, res in zip(group, results):
                text = res.outputs[0].text
                obj = parse_json_object(text)
                out.append(prediction(r, self.name, variant, obj, raw=text,
                                      error=None if obj is not None else "unparseable output",
                                      latency_s=per_case))
        return out


# ------------------------------------------------------------------ NuExtract


def nuextract_template(calculator: str, variant: str) -> Dict:
    template = {}
    for f in schemas.field_specs(calculator, variant):
        if f.kind == "bool":
            template[f.name] = ["true", "false"]
        elif f.kind == "enum":
            template[f.name] = list(f.choices)
        elif f.kind == "int":
            template[f.name] = "integer"
        else:
            template[f.name] = "number"
    return template


def nuextract_json_schema(calculator: str, variant: str) -> Dict:
    """JSON schema of NuExtract's output for ``nuextract_template`` (for constrained decoding)."""
    props = {}
    for f in schemas.field_specs(calculator, variant):
        if f.kind == "bool":
            prop = {"enum": ["true", "false", None]}
        elif f.kind == "enum":
            prop = {"enum": list(f.choices) + [None]}
        elif f.kind == "int":
            prop = {"type": ["integer", "null"]}
        else:
            prop = {"type": ["number", "null"]}
        props[f.name] = prop
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def nuextract_instructions(calculator: str, variant: str) -> str:
    absent = "" if variant == "R" else (
        " Answer false for a yes/no criterion the text does not mention and 0 for a dose that is not given."
    )
    return f"Calculator: {schemas.CALCULATOR_NAMES[calculator]}.{absent}\n" + schemas.field_guide(calculator, variant)


def from_nuextract(calculator: str, variant: str, obj: Optional[Dict]) -> Optional[Dict]:
    """Map NuExtract's template-typed output onto the request schema."""
    if obj is None:
        return None
    out = {}
    for f in schemas.field_specs(calculator, variant):
        v = obj.get(f.name)
        # the model sometimes wraps a value in a list or returns another shape; keep a single
        # scalar, otherwise treat the field as unanswered
        if isinstance(v, list):
            v = v[0] if len(v) == 1 else None
        if isinstance(v, dict):
            v = None
        if f.kind == "bool":
            v = {"true": True, "false": False, True: True, False: False}.get(v, None)
            if v is None and not f.optional:
                v = False
        elif f.kind in ("int", "float") and v is None and not f.optional:
            v = 0
        out[f.name] = v
    return out


class NuExtractExtractor(Extractor):
    def __init__(self, model: str, name: Optional[str] = None, few_shot: int = 0,
                 train_records: Optional[List[Dict]] = None, max_tokens: int = 1024,
                 instructions: bool = True, enable_thinking: Optional[bool] = False, constrained: bool = False,
                 backend: str = "vllm", **engine_kwargs):
        self.name = name or (f"nuextract:{os.path.basename(model.rstrip('/'))}" + (f":{few_shot}shot" if few_shot else "")
                             + (":constrained" if constrained else ""))
        self.constrained = constrained
        self.backend = backend
        if backend == "transformers":
            # for laptops and CPU pilots; no constrained decoding on this path
            if constrained:
                raise ValueError("--constrained needs the vllm backend")
            self._load_transformers(model)
        else:
            engine_kwargs.setdefault("trust_remote_code", True)
            engine_kwargs.setdefault("limit_mm_per_prompt", {"image": 0})
            self.llm = load_vllm(model, **engine_kwargs)
        self.few_shot = few_shot
        self.train = train_records or []
        self.max_tokens = max_tokens
        self.instructions = instructions
        self.enable_thinking = enable_thinking

    def _load_transformers(self, model: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, padding_side="left")
        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            model, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None)
        self.hf_model.eval()

    def _generate_transformers(self, conversations, template_kwargs) -> List[str]:
        import torch

        texts = [self.processor.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                              **template_kwargs) for c in conversations]
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.hf_model.device)
        with torch.inference_mode():
            out = self.hf_model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=self.max_tokens)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _template_kwargs(self, calc: str, variant: str) -> Dict:
        kwargs = {"template": json.dumps(nuextract_template(calc, variant), indent=4)}
        if self.instructions:
            kwargs["instructions"] = nuextract_instructions(calc, variant)
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        examples = few_shot_examples(self.train, calc, variant, self.few_shot)
        if examples:
            kwargs["examples"] = [
                {"input": ex["vignette"], "output": json.dumps(_to_nuextract_output(calc, variant, ex["gold"][variant]))}
                for ex in examples
            ]
        return kwargs

    def extract_batch(self, records, variant):
        out = []
        by_calc: Dict[str, List[Dict]] = {}
        for r in records:
            by_calc.setdefault(r["calculator"], []).append(r)
        for calc, group in by_calc.items():
            conversations = [[{"role": "user", "content": [{"type": "text", "text": r["vignette"]}]}] for r in group]
            start = time.perf_counter()
            if self.backend == "transformers":
                texts = self._generate_transformers(conversations, self._template_kwargs(calc, variant))
            else:
                # NuExtract is used natively by default; --constrained adds JSON-schema decoding
                params = _sampling_params(nuextract_json_schema(calc, variant) if self.constrained else None,
                                          self.max_tokens)
                results = self.llm.chat(conversations, sampling_params=params, use_tqdm=False,
                                        chat_template_kwargs=self._template_kwargs(calc, variant))
                texts = [res.outputs[0].text for res in results]
            per_case = (time.perf_counter() - start) / max(len(group), 1)
            for r, text in zip(group, texts):
                # NuExtract3 in thinking mode emits reasoning before the JSON
                obj = parse_json_object(text.split("</think>")[-1])
                out.append(prediction(r, self.name, variant, from_nuextract(calc, variant, obj), raw=text,
                                      error=None if obj is not None else "unparseable output",
                                      latency_s=per_case))
        return out


def _to_nuextract_output(calculator: str, variant: str, gold: Dict) -> Dict:
    out = {}
    for f in schemas.field_specs(calculator, variant):
        v = gold.get(f.name)
        out[f.name] = ("true" if v else "false") if (f.kind == "bool" and v is not None) else v
    return out


# ------------------------------------------------------------------ OpenAI-compatible APIs


class OpenAICompatExtractor(Extractor):
    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY",
                 name: Optional[str] = None, few_shot: int = 0, train_records: Optional[List[Dict]] = None,
                 max_tokens: int = 1024, concurrency: int = 8, timeout_s: float = 120):
        self.name = name or f"api:{model}" + (f":{few_shot}shot" if few_shot else "")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.few_shot, self.train = few_shot, train_records or []
        self.max_tokens, self.concurrency, self.timeout_s = max_tokens, concurrency, timeout_s

    def _call(self, record: Dict, variant: str) -> Dict:
        calc = record["calculator"]
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": chat_messages(record, variant, few_shot_examples(self.train, calc, variant, self.few_shot)),
            "response_format": {"type": "json_schema", "json_schema": {
                "name": f"{calc}_request", "schema": schemas.json_schema(calc, variant)}},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                text = json.load(resp)["choices"][0]["message"]["content"] or ""
            obj = parse_json_object(text)
            return prediction(record, self.name, variant, obj, raw=text,
                              error=None if obj is not None else "unparseable output",
                              latency_s=time.perf_counter() - start)
        except Exception as e:  # noqa: BLE001 - recorded per case, the run continues
            return prediction(record, self.name, variant, None, error=f"request failed: {e}",
                              latency_s=time.perf_counter() - start)

    def extract_batch(self, records, variant):
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(lambda r: self._call(r, variant), records))
