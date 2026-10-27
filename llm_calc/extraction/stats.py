"""Small, dependency-free statistics for the benchmark (numpy only)."""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def wilson(successes: int, n: int, z: float = 1.959964) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict:
    """Exact two-sided McNemar test on paired binary outcomes."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_only": b, "b_only": c, "p": min(1.0, 2 * tail)}


def bootstrap_ci(values_by_cluster: List[Sequence[float]], stat: Callable[[List[Sequence[float]]], float],
                 n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Percentile CI, resampling clusters (cases) with replacement."""
    rng = np.random.default_rng(seed)
    n = len(values_by_cluster)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(stat([values_by_cluster[i] for i in idx]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out
