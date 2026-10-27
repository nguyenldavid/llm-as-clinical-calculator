"""
Common interface for extraction systems.

An extractor turns benchmark records into predictions. Batch-oriented, because
the GPU systems (encoders, NuExtract, vLLM) are far faster on batches.
"""

import json
import re
import time
from typing import Dict, Iterable, List, Optional

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Extractor:
    name: str = "base"

    def extract_batch(self, records: List[Dict], variant: str) -> List[Dict]:
        raise NotImplementedError

    def run(self, records: Iterable[Dict], variant: str, batch_size: int = 64) -> List[Dict]:
        records = list(records)
        out = []
        for i in range(0, len(records), batch_size):
            out += self.extract_batch(records[i : i + batch_size], variant)
        return out


def prediction(record: Dict, system: str, variant: str, output: Optional[Dict], raw: str = "",
               error: Optional[str] = None, latency_s: Optional[float] = None) -> Dict:
    return {"id": record["id"], "system": system, "variant": variant, "output": output,
            "raw": raw, "error": error, "latency_s": latency_s}


def parse_json_object(text: str) -> Optional[Dict]:
    """First JSON object in model output (tolerates code fences and preambles)."""
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    m = JSON_OBJECT.search(text or "")
    if not m:
        return None
    try:
        value = json.loads(m.group())
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
