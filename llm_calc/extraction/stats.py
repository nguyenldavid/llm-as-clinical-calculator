"""
Statistics for the benchmark.

Resampling unit
---------------
Rows are not independent. A case contributes several fields; a T1 case reappears
in T4 (perturbed) and T3 (paraphrased); and every system sees the same cases.
Every interval here therefore comes from a *cluster* bootstrap that resamples
whole clusters (``evaluate`` sets ``cluster`` to the base case id), and every
comparison is paired on identical rows.

The bootstrap is vectorised: each resample is a vector of multinomial cluster
weights, so any estimate that is a weighted combination of cell means (micro
accuracy, macro accuracy, a paired difference, Brier score, ECE) is two matrix
products per batch of resamples.

Everything except ``gee_field_model`` (statsmodels) needs numpy only.
"""

import math
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

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


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


N_BOOT = 2000
Z95 = 1.959964


def _cluster_matrix(clusters: Sequence[Hashable], cells: Sequence[Hashable], values: np.ndarray):
    """Per-cluster sums and counts in each cell: (n_clusters, n_cells) arrays."""
    c_codes, c_index = _codes(clusters)
    k_codes, k_index = _codes(cells)
    sums = np.zeros((len(c_index), len(k_index)))
    counts = np.zeros_like(sums)
    np.add.at(sums, (c_codes, k_codes), values)
    np.add.at(counts, (c_codes, k_codes), 1.0)
    return sums, counts, k_index


def _codes(keys: Sequence[Hashable]):
    index: Dict[Hashable, int] = {}
    codes = np.fromiter((index.setdefault(k, len(index)) for k in keys), dtype=np.int64, count=len(keys))
    return codes, list(index)


def _multinomial_weights(n_clusters: int, n_boot: int, rng: np.random.Generator, batch: int = 250):
    for start in range(0, n_boot, batch):
        size = min(batch, n_boot - start)
        yield rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=size).astype(np.float64)


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[Hashable], cells: Optional[Sequence[Hashable]] = None,
                      cell_weights: Optional[Dict[Hashable, float]] = None, n_boot: int = N_BOOT, seed: int = 0) -> Dict:
    """Estimate and 95% percentile CI of a weighted mean of cell means.

    ``cells=None`` gives the plain (micro) mean. For macro accuracy pass the
    (calculator, field) of each row as its cell and weights that average fields
    within a calculator and then calculators (``macro_weights``). A paired
    difference is the same call on per-row differences of two systems.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_clusters": 0}
    cells = cells if cells is not None else [0] * len(v)
    sums, counts, cell_index = _cluster_matrix(clusters, cells, v)
    w = np.array([1.0 if cell_weights is None else cell_weights[k] for k in cell_index])
    w = w / w.sum()

    def estimate(cluster_w: np.ndarray) -> np.ndarray:
        s, c = cluster_w @ sums, cluster_w @ counts
        with np.errstate(invalid="ignore", divide="ignore"):
            means = s / c
        # a resample can miss a cell entirely; average over the cells it has
        present = ~np.isnan(means)
        return np.nansum(means * w, axis=-1) / (present * w).sum(axis=-1)

    est = float(estimate(np.ones(len(sums))))
    rng = np.random.default_rng(seed)
    draws = np.concatenate([estimate(W) for W in _multinomial_weights(len(sums), n_boot, rng)])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"est": est, "lo": float(lo), "hi": float(hi), "n": len(v), "n_clusters": len(sums)}


def macro_weights(cells: Sequence[Tuple[Hashable, Hashable]]) -> Dict[Tuple, float]:
    """Cell weights for "mean over fields within calculator, then over calculators"."""
    per_calc: Dict[Hashable, set] = {}
    for calc, field in set(cells):
        per_calc.setdefault(calc, set()).add(field)
    return {(calc, f): 1.0 / (len(per_calc) * len(fs)) for calc, fs in per_calc.items() for f in fs}


def paired(a: Sequence[bool], b: Sequence[bool], clusters: Sequence[Hashable], cells=None, cell_weights=None,
           n_boot: int = N_BOOT, seed: int = 0) -> Dict:
    """Paired difference a - b with a cluster-bootstrap CI, plus the exact McNemar test on the rows."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = cluster_bootstrap(diff, clusters, cells, cell_weights, n_boot, seed)
    out.update(mcnemar_exact(a, b))
    return out


# ---------------------------------------------------------------- calibration


def calibration(confidence: Sequence[float], correct: Sequence[bool], clusters: Sequence[Hashable],
                n_bins: int = 10, n_boot: int = N_BOOT, seed: int = 0) -> Dict:
    """Brier score, ECE (equal-width bins), AUROC and AURC, with cluster-bootstrap CIs for Brier and ECE.

    ``confidence`` must be the system's probability that its answer is right
    (for Jev: the probability of the option it chose).
    """
    p = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    sq = (p - y) ** 2
    gap = p - y
    c_codes, c_index = _codes(clusters)
    n_c = len(c_index)
    sq_s = np.bincount(c_codes, weights=sq, minlength=n_c)
    cnt = np.bincount(c_codes, minlength=n_c).astype(float)
    gap_s = np.zeros((n_c, n_bins))
    np.add.at(gap_s, (c_codes, bins), gap)

    def brier(W):
        return (W @ sq_s) / (W @ cnt)

    def ece(W):
        return np.abs(W @ gap_s).sum(axis=-1) / (W @ cnt)

    ones = np.ones(n_c)
    rng = np.random.default_rng(seed)
    b_draws, e_draws = [], []
    for W in _multinomial_weights(n_c, n_boot, rng):
        b_draws.append(brier(W))
        e_draws.append(ece(W))
    b_draws, e_draws = np.concatenate(b_draws), np.concatenate(e_draws)
    reliability = []
    for k in range(n_bins):
        m = bins == k
        if m.any():
            reliability.append({"bin": k, "lo": k / n_bins, "hi": (k + 1) / n_bins, "n": int(m.sum()),
                                "mean_confidence": float(p[m].mean()), "accuracy": float(y[m].mean())})
    return {
        "n": len(p), "accuracy": float(y.mean()), "mean_confidence": float(p.mean()),
        "brier": float(brier(ones)), "brier_lo": float(np.percentile(b_draws, 2.5)),
        "brier_hi": float(np.percentile(b_draws, 97.5)),
        "ece": float(ece(ones)), "ece_lo": float(np.percentile(e_draws, 2.5)),
        "ece_hi": float(np.percentile(e_draws, 97.5)),
        "auroc": auroc(p, y), "aurc": aurc(p, y), "reliability": reliability,
    }


def auroc(score: Sequence[float], label: Sequence[float]) -> float:
    """Probability that a correct answer has higher confidence than a wrong one (ties count half)."""
    s, y = np.asarray(score, dtype=float), np.asarray(label, dtype=bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    # average ranks over ties
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aurc(confidence: Sequence[float], correct: Sequence[float]) -> float:
    """Area under the risk-coverage curve (lower is better), deferring the least confident first."""
    p, y = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    if len(p) == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    risk = np.cumsum(1 - y[order]) / np.arange(1, len(p) + 1)
    return float(risk.mean())


# ---------------------------------------------------------------- power


def _norm_ppf(q: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, |error| < 1.2e-9 after one Newton step)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    if q < 0.02425:
        t = math.sqrt(-2 * math.log(q))
        x = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
            ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    elif q > 1 - 0.02425:
        return -_norm_ppf(1 - q)
    else:
        t = q - 0.5
        r = t * t
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - q
    return x - e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)


def mde_paired(n: int, discordance: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest accuracy difference a paired (McNemar) test detects.

    Connor (1987): n = [z_{a/2} sqrt(psi) + z_b sqrt(psi - d^2)]^2 / d^2, where psi
    is the share of discordant pairs; solved for d by bisection.
    """
    za, zb = _norm_ppf(1 - alpha / 2), _norm_ppf(power)

    def needed(d):
        return (za * math.sqrt(discordance) + zb * math.sqrt(max(discordance - d * d, 0.0))) ** 2 / (d * d)

    lo, hi = 1e-6, math.sqrt(discordance)
    if needed(hi) > n:
        return float("nan")  # not detectable at this discordance
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if needed(mid) > n else (lo, mid)
    return hi


def ci_half_width(p: float, n_eff: float) -> float:
    return Z95 * math.sqrt(p * (1 - p) / n_eff)


# ---------------------------------------------------------------- model


def gee_field_model(rows, reference_system: str, reference_set: str = "T1", working: str = "independence") -> Dict:
    """Population-averaged logistic model of field correctness.

    correct ~ system * test_set + field kind + calculator, with a robust
    (sandwich) covariance clustered by base case, so the copies of a case in T1,
    T3 and T4 and its answers from every system form one cluster. Returns the
    coefficient table (odds ratios) and joint Wald tests per term; the
    system x test_set interaction is the formal test of robustness degradation
    (H1, H2).

    ``working="exchangeable"`` is the sensitivity analysis (S6): the robust SEs
    are valid under either working correlation, so the conclusions should agree.

    ``rows`` is a DataFrame with correct, system, test_set, kind, calculator,
    cluster. Requires statsmodels.
    """
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = rows[["correct", "system", "test_set", "kind", "calculator", "cluster"]].copy()
    df["correct"] = df["correct"].astype(float)
    df = df.sort_values("cluster", kind="mergesort")
    formula = (f"correct ~ C(system, Treatment({reference_system!r})) * C(test_set, Treatment({reference_set!r})) "
               "+ C(kind, Treatment('bool')) + C(calculator)")
    model = smf.gee(formula, groups="cluster", data=df, family=sm.families.Binomial(),
                    cov_struct={"independence": sm.cov_struct.Independence(),
                                "exchangeable": sm.cov_struct.Exchangeable()}[working])
    fit = model.fit()
    ci = fit.conf_int()
    table = pd.DataFrame({"term": fit.params.index, "log_odds": fit.params.values, "se": fit.bse.values,
                          "or": np.exp(fit.params.values), "or_lo": np.exp(ci[0].values),
                          "or_hi": np.exp(ci[1].values), "p": fit.pvalues.values})
    wald = fit.wald_test_terms(skip_single=False, scalar=True)
    terms = pd.DataFrame({"term": wald.table.index, "chi2": wald.table["statistic"].astype(float).values,
                          "df": wald.table["df_constraint"].values, "p": wald.table["pvalue"].astype(float).values})
    return {"coefficients": table, "terms": terms, "n_rows": len(df), "n_clusters": df.cluster.nunique()}
