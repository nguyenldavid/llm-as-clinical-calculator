"""
Figures and tables for the extraction benchmark.

One code path for mock and real results: point it at an evaluation directory
(``cases.jsonl`` and ``fields.jsonl`` from ``evaluate`` or ``mock-results``;
optional ``error_codes.tsv``, ``paper_arms.tsv``, ``systems.tsv``) and it writes
PNG figures plus a markdown page with every table. With ``--mock`` every figure
carries a watermark and every table a MOCK label.

Figures that already have real data behind them (dataset table, Abacus-409
attribution, CPU baselines) are always drawn from the real files.

    python -m llm_calc.extraction figures --eval-dir DIR --out docs/mock_results --mock

Styling follows the repository's chart conventions: thin marks, hairline grid,
one fixed categorical order per system family, sequential blue for magnitude,
blue/red with a grey midpoint for signed change, a legend whenever there are two
or more series, and a table twin for every figure.
"""

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from llm_calc.extraction import datasets as ds
from llm_calc.extraction import stats

# ---------------------------------------------------------------- palette (light)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7",
        400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
        700: "#0d366b"}
NEUTRAL_MID = "#f0efec"
RED_ARM = ["#f6c5c4", "#ec8a89", "#e34948", "#b8302f"]

FAMILIES = ["Rules", "Encoder", "NuExtract", "Open LLM", "API LLM", "System One"]
FAMILY_COLOR = dict(zip(FAMILIES, CATEGORICAL))
SET_ORDER = ["T1", "A409", "T2", "T3", "T4", "T5"]
PRIMARY = ("T2", "T3")
SET_LABEL = {"T1": "T1 template", "A409": "A409 paper failures", "T2": "T2 held-out phrasing",
             "T3": "T3 paraphrase", "T4": "T4 perturbed", "T5": "T5 MedCalc-Bench"}
PERTURBATIONS = ["distractors", "units", "conflicting", "abbreviations", "reorder", "long_note", "injection",
                 "omission"]
# journal column widths (Nature/Cell family: 89 mm single, 183 mm double)
SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Inter for text, DejaVu Sans as the fallback for glyphs Inter lacks (∪, Δ)
    family = ["Inter", "DejaVu Sans"] if any("Inter" in f.name for f in font_manager.fontManager.ttflist) \
        else ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    })
    return plt


class Page:
    """Collects figures and tables into one markdown document."""

    def __init__(self, out: Path, mock: bool):
        self.out, self.mock = out, mock
        self.fig_dir = out / "figures"
        self.data_dir = out / "source_data"
        for d in (self.fig_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.parts: List[str] = []

    def label(self, real: bool, shell: bool = False) -> str:
        if shell:
            return "SHELL"
        return "REAL" if real else ("MOCK" if self.mock else "")

    def figure(self, fig, key: str, title: str, caption: str, real: bool = False,
               data: Optional[pd.DataFrame] = None):
        """Save a figure as PDF (vector, for submission) and PNG (for this page), plus its source data.

        Numbers are assigned in page order when the page is written; captions
        refer to other items as ``{fig:key}`` / ``{tab:key}``.
        """
        plt = _plt()
        if self.mock and not real:
            fig.text(0.5, 0.5, "MOCK DATA · ILLUSTRATIVE ONLY", ha="center", va="center", rotation=24,
                     fontsize=30, color=MUTED, alpha=0.16, weight="bold", zorder=100)
        fig.savefig(self.fig_dir / f"{key}.png", dpi=300, bbox_inches="tight")
        fig.savefig(self.fig_dir / f"{key}.pdf", bbox_inches="tight")
        plt.close(fig)
        source = ""
        if data is not None:
            data.to_csv(self.data_dir / f"{key}.tsv", sep="\t", index=False, float_format="%.6g")
            source = f" Source data: `source_data/{key}.tsv`."
        tag = self.label(real)
        self.parts.append(f"### {{figdef:{key}}}. {title}{f' ({tag})' if tag else ''}\n\n"
                          f"![{title}](figures/{key}.png)\n\n{caption}{source}\n")

    def table(self, df: pd.DataFrame, key: str, title: str, caption: str = "", real: bool = False,
              shell: bool = False):
        tag = self.label(real, shell)
        if not shell:
            df.to_csv(self.data_dir / f"table_{key}.tsv", sep="\t", index=False, float_format="%.6g")
        self.parts.append(f"### {{tabdef:{key}}}. {title}{f' ({tag})' if tag else ''}\n\n{_markdown(df)}\n\n"
                          f"{caption}\n")

    def text(self, md: str):
        self.parts.append(md)

    def write(self, header: str) -> Path:
        body = "\n".join(self.parts)
        numbers: Dict[str, int] = {}
        counters = {"fig": 0, "tab": 0}
        for kind, key in re.findall(r"\{(fig|tab)def:([a-z0-9_]+)\}", body):
            counters[kind] += 1
            numbers[f"{kind}:{key}"] = counters[kind]
        word = {"fig": "Figure", "tab": "Table"}

        def resolve(m):
            kind, key = m.group(1), m.group(3)
            n = numbers.get(f"{kind}:{key}")
            return f"{word[kind]} {n}" if n else f"{word[kind]} (not rendered)"

        body = re.sub(r"\{(fig|tab)(def)?:([a-z0-9_]+)\}", resolve, body)
        path = self.out / "README.md"
        path.write_text(header + "\n\n" + body)
        return path


def _markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def _panel_letters(axes):
    """Bold lower-case panel letters at the top left of each axis (journal convention)."""
    for letter, ax in zip("abcdefgh", axes):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def _pct(axis):
    from matplotlib.ticker import PercentFormatter

    axis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _short(system: str) -> str:
    if system.startswith("jev:"):
        return "Jev " + system.split("jev-")[-1].rsplit(".0", 1)[0]
    return (system.replace("vllm:", "").replace("nuextract:", "").replace("encoder-", "").replace("api:", "")
            .replace("-Instruct", "").replace(":3shot", " (3-shot)").replace(":constrained", " (constrained)"))


# ---------------------------------------------------------------- data


def load(eval_dir: Path):
    cases = pd.read_json(eval_dir / "cases.jsonl", lines=True)
    fields = pd.read_json(eval_dir / "fields.jsonl", lines=True)
    systems_file = eval_dir / "systems.tsv"
    if systems_file.exists():
        systems = pd.read_csv(systems_file, sep="\t")
    else:
        systems = pd.DataFrame({"system": sorted(cases.system.unique())})
        systems["family"] = systems.system.map(_guess_family)
        systems["deployment"] = systems.family.map(
            {"Rules": "on-prem CPU", "API LLM": "external API"}).fillna("on-prem GPU")
    fam = dict(zip(systems.system, systems.family))
    for df in (cases, fields):
        df["family"] = df.system.map(fam).fillna("Other")
        if "perturbation" not in df:
            df["perturbation"] = ""
        df["perturbation"] = df["perturbation"].fillna("")
    # the clinician reference arm is not a system family: it is drawn as a reference and compared separately
    clin = {"cases": cases[cases.family == "Clinician"], "fields": fields[fields.family == "Clinician"]}
    cases = cases[cases.family.isin(FAMILIES)]
    fields = fields[fields.family.isin(FAMILIES)]
    return cases, fields, systems, clin


def _guess_family(system: str) -> str:
    if system == "rules":
        return "Rules"
    for prefix, fam in [("encoder", "Encoder"), ("nuextract", "NuExtract"), ("vllm", "Open LLM"), ("api", "API LLM"),
                        ("jev", "System One"), ("clinician", "Clinician")]:
        if system.startswith(prefix):
            return fam
    return "Other"


def _system_order(cases: pd.DataFrame) -> List[str]:
    """Systems grouped by family, best primary score first within a family."""
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    if d.empty:
        d = cases[cases.variant == "D"]
    acc = d.groupby("system").score_correct.mean()
    fam = cases.drop_duplicates("system").set_index("system").family
    order = []
    for f in FAMILIES:
        members = [s for s in acc.index if fam.get(s) == f]
        order += sorted(members, key=lambda s: -acc[s])
    return order


def _wilson(k: int, n: int):
    lo, hi = stats.wilson(int(k), int(n))
    return (lo if lo is not None else float("nan")), (hi if hi is not None else float("nan"))


def _clusters(df: pd.DataFrame) -> pd.Series:
    return df["cluster"] if "cluster" in df else df["id"]


def _boot(df: pd.DataFrame, col: str = "score_correct") -> Dict:
    """Cluster-bootstrap mean and 95% CI of a case-level column."""
    return stats.cluster_bootstrap(df[col].astype(float).values, _clusters(df).tolist())


def _macro_boot(f: pd.DataFrame) -> Dict:
    """Macro field accuracy (fields within calculator, then calculators) with a cluster-bootstrap CI."""
    cells = list(zip(f.calculator, f.field))
    return stats.cluster_bootstrap(f.correct.astype(float).values, _clusters(f).tolist(), cells,
                                   stats.macro_weights(cells))


def _est_ci(r: Dict, digits: int = 3) -> str:
    if r["n"] == 0 or math.isnan(r["est"]):
        return "—"
    return f"{r['est']:.{digits}f} ({r['lo']:.{digits}f} to {r['hi']:.{digits}f})"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _paired_cases(cases: pd.DataFrame, a: str, b: str, col: str = "score_correct") -> pd.DataFrame:
    wide = cases[cases.system.isin([a, b])].pivot_table(index=["id", "cluster"], columns="system", values=col,
                                                         aggfunc="first")
    return wide.dropna().reset_index()


def _paired_fields(fields: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    wide = fields[fields.system.isin([a, b])].pivot_table(
        index=["id", "cluster", "calculator", "field"], columns="system", values="correct", aggfunc="first")
    return wide.dropna().reset_index()


def compare(cases: pd.DataFrame, fields: pd.DataFrame, a: str, b: str) -> Dict:
    """Paired a - b on final-score accuracy and macro field accuracy, same cases and fields."""
    pc = _paired_cases(cases, a, b)
    score = stats.paired(pc[a].astype(bool).values, pc[b].astype(bool).values, pc.cluster.tolist())
    pf = _paired_fields(fields, a, b)
    cells = list(zip(pf.calculator, pf.field))
    macro = stats.cluster_bootstrap(pf[a].astype(float).values - pf[b].astype(float).values, pf.cluster.tolist(),
                                    cells, stats.macro_weights(cells))
    return {"n_cases": len(pc), "score": score, "macro": macro}


# ---------------------------------------------------------------- figures


def fig_headline(page: Page, cases: pd.DataFrame, clin: Optional[Dict] = None):
    plt = _plt()
    d = cases[cases.variant == "D"]
    order = _system_order(cases)
    sets = [s for s in SET_ORDER if s in set(d.test_set)]
    clin_cases = clin["cases"] if clin is not None else pd.DataFrame(columns=["test_set"])
    fig, axes = plt.subplots(1, len(sets), figsize=(DOUBLE_COL, 0.18 * len(order) + 1.3), sharey=True)
    y = {s: i for i, s in enumerate(reversed(order))}
    rows = []
    for ax, test_set in zip(axes, sets):
        for (system, family), sub in d[d.test_set == test_set].groupby(["system", "family"]):
            r = _boot(sub)
            rows.append({"system": system, "family": family, "test_set": test_set, "n": r["n"],
                         "n_clusters": r["n_clusters"], "accuracy": r["est"], "lo": r["lo"], "hi": r["hi"]})
            ax.plot([r["lo"], r["hi"]], [y[system]] * 2, color=AXIS, lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter(r["est"], y[system], s=34, color=FAMILY_COLOR[family], edgecolor=SURFACE, linewidth=1.6,
                       zorder=3)
        ref = clin_cases[clin_cases.test_set == test_set]
        if len(ref):
            ax.axvline(ref.score_correct.mean(), color=INK, lw=1, ls=(0, (3, 2)), zorder=1)
            rows.append({"system": "clinicians (pooled)", "family": "Clinician", "test_set": test_set,
                         "n": len(ref), "accuracy": ref.score_correct.mean()})
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50", "100%"])
        ax.set_title(SET_LABEL[test_set].replace(" ", "\n", 1), fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels([_short(s) for s in y])
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], markersize=5, label=f) for f in FAMILIES]
    if len(clin_cases):
        handles.append(plt.Line2D([], [], color=INK, lw=1, ls=(0, (3, 2)), label="Clinicians"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    fig.supxlabel("Final-score accuracy (OpenMedCalc on the extracted request), 95% cluster-bootstrap CI",
                  fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    page.figure(fig, "headline", "Final-score accuracy by system and test set",
                "Primary endpoint 1 (plan §8.1). Rows are grouped by system family and sorted by accuracy on the "
                "primary sets (T2 ∪ T3). T1 and A409 are reported for comparability only (plan §5.3). "
                "Intervals resample whole base cases (2,000 resamples), so perturbed and paraphrased copies of a "
                "case move together. Dashed line: the two clinicians' pooled accuracy on their stratified sample "
                "of that set ({tab:clinician}). {tab:primary} is the table view.", data=pd.DataFrame(rows))


def fig_generalisation(page: Page, cases: pd.DataFrame):
    plt = _plt()
    d = cases[cases.variant == "D"]
    order = _system_order(cases)
    t1 = d[d.test_set == "T1"].groupby("system").score_correct.mean()
    prim = d[d.test_set.isin(PRIMARY)].groupby("system").score_correct.mean()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.27 * len(order) + 1.2))
    for i, s in enumerate(reversed(order)):
        if s not in t1 or s not in prim:
            continue
        ax.plot([prim[s], t1[s]], [i, i], color=BLUE[200], lw=2, solid_capstyle="round", zorder=1)
        ax.scatter(t1[s], i, s=34, color=BLUE[250], edgecolor=SURFACE, linewidth=1.6, zorder=2)
        ax.scatter(prim[s], i, s=34, color=BLUE[600], edgecolor=SURFACE, linewidth=1.6, zorder=3)
        ax.text(1.02, i, f"−{100 * (t1[s] - prim[s]):.0f} pp", va="center", fontsize=7.5, color=INK_2,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_short(s) for s in reversed(order)])
    ax.set_xlim(0, 1)
    _pct(ax.xaxis)
    ax.set_xlabel("Final-score accuracy")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE[250], label="T1 (template text)"),
                       plt.Line2D([], [], marker="o", ls="", color=BLUE[600], label="T2 ∪ T3 (new wording)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    page.figure(fig, "generalisation", "Generalisation gap: template text vs new wording",
                "H1/H2 (plan §3). The label is the drop in percentage points from T1 to the primary sets. "
                "Systems that memorise phrasings (rules written on T1, fine-tuned encoders) are expected to show "
                "the largest drops.")


def fig_heatmap(page: Page, fields: pd.DataFrame, cases: pd.DataFrame):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    f = fields[(fields.variant == "D") & fields.test_set.isin(PRIMARY)]
    if f.empty:
        return
    order = _system_order(cases)
    calcs = sorted(f.calculator.unique())
    per = f.groupby(["system", "calculator", "field"]).correct.mean().groupby(["system", "calculator"]).mean()
    mat = per.unstack("calculator").reindex(index=order, columns=calcs)
    cmap = LinearSegmentedColormap.from_list("seq", [BLUE[100], BLUE[300], BLUE[500], BLUE[700]])
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.2 * len(order) + 1.3))
    im = ax.imshow(mat.values, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(calcs)))
    ax.set_xticklabels(calcs, rotation=40, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_short(s) for s in order])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="min")
    cb.outline.set_visible(False)
    _pct(cb.ax.yaxis)
    cb.set_label("Macro field accuracy (T2 ∪ T3); below 50% shown as 50%", color=INK_2)
    fig.tight_layout()
    page.figure(fig, "field_heatmap", "Field accuracy by system and calculator",
                "Primary endpoint 2 (plan §8.1), broken down by calculator. Scale starts at 50%. "
                "{tab:per_calculator} gives the per-calculator numbers for the best system of each family.")


def fig_t4(page: Page, cases: pd.DataFrame):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    d = cases[cases.variant == "D"]
    t1 = d[d.test_set == "T1"].set_index(["system", "id"]).score_correct
    t4 = d[d.test_set == "T4"].copy()
    if t4.empty:
        return
    t4["base"] = t4.id.str.replace(r"^T4-[a-z_]+-", "", regex=True)
    t4["t1_correct"] = [t1.get((s, b), float("nan")) for s, b in zip(t4.system, t4.base)]
    t4["delta"] = t4.score_correct.astype(float) - t4.t1_correct
    mat = t4.groupby(["family", "perturbation"]).delta.mean().unstack().reindex(
        index=FAMILIES, columns=[p for p in PERTURBATIONS if p in set(t4.perturbation)]) * 100
    cmap = LinearSegmentedColormap.from_list("div", [RED_ARM[3], RED_ARM[1], NEUTRAL_MID, BLUE[300], BLUE[600]])
    lim = max(10, float(abs(mat.values[~pd.isna(mat.values)]).max()))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.42 * len(FAMILIES) + 0.8))
    im = ax.imshow(mat.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in mat.columns])
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels(FAMILIES)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Δ score accuracy vs same case in T1 (pp)", color=INK_2)
    fig.tight_layout()
    page.figure(fig, "t4_robustness", "Robustness to controlled perturbations (T4)",
                "RQ4. Each cell is the paired change in final-score accuracy, T4 minus the same base case in T1, "
                "averaged over the systems in a family. Red is worse than on the original text.")


def fig_field_kind(page: Page, fields: pd.DataFrame):
    plt = _plt()
    import numpy as np

    f = fields[(fields.variant == "D") & fields.test_set.isin(PRIMARY)]
    kinds = ["bool", "enum", "int", "float"]
    acc = f.groupby(["family", "kind"]).correct.mean().unstack().reindex(index=FAMILIES, columns=kinds)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
    width = 0.8 / len(FAMILIES) - 0.01
    x = np.arange(len(kinds))
    for i, fam in enumerate(FAMILIES):
        ax.bar(x + (i - (len(FAMILIES) - 1) / 2) * (width + 0.01), acc.loc[fam].values, width=width,
               color=FAMILY_COLOR[fam],
               label=fam, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["yes/no", "category (enum)", "integer", "decimal"])
    ax.set_ylim(0, 1.0)
    _pct(ax.yaxis)
    ax.set_ylabel("Field accuracy (T2 ∪ T3)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(FAMILIES), loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    page.figure(fig, "field_kind", "Accuracy by field type",
                "Secondary endpoint (plan §8.2). Enum fields include ordinal scales "
                "(NIHSS, GAD-7), where off-by-one readings are the expected error.")


def fig_assertion(page: Page, fields: pd.DataFrame):
    plt = _plt()
    f = fields[(fields.variant == "D") & (fields.test_set == "T5")].copy()
    if f.empty:
        return
    f["documented"] = f.source != "not documented"
    acc = f.groupby(["family", "documented"]).correct.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, fam in enumerate(reversed(FAMILIES)):
        if fam not in acc.index:
            continue
        a, b = acc.loc[fam].get(True), acc.loc[fam].get(False)
        ax.plot([a, b], [i, i], color=BLUE[200], lw=2, zorder=1)
        ax.scatter(a, i, s=34, color=BLUE[600], edgecolor=SURFACE, linewidth=1.6, zorder=2)
        ax.scatter(b, i, s=34, color=BLUE[250], edgecolor=SURFACE, linewidth=1.6, zorder=2)
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels(list(reversed(FAMILIES)))
    ax.set_xlim(0.4, 1.0)
    _pct(ax.xaxis)
    ax.set_xlabel("Field accuracy on T5 (MedCalc-Bench)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE[600], label="stated in the note"),
                       plt.Line2D([], [], marker="o", ls="", color=BLUE[250], label="not documented (gold = absent)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    page.figure(fig, "assertion", "Documented vs undocumented criteria (T5)",
                "Plan §8.3. Undocumented criteria test whether a system invents findings (hallucinated presence). "
                "Axis starts at 40%.")


def fig_variant(page: Page, cases: pd.DataFrame):
    plt = _plt()
    d = cases[(cases.calculator == "hasbled") & cases.test_set.isin(PRIMARY)]
    acc = d.groupby(["system", "variant"]).score_correct.mean().unstack()
    if "R" not in acc:
        return
    order = [s for s in _system_order(cases) if s in acc.index]
    fig, ax = plt.subplots(figsize=(6.8, 0.27 * len(order) + 1.2))
    for i, s in enumerate(reversed(order)):
        dv, rv = acc.loc[s, "D"], acc.loc[s, "R"]
        ax.plot([dv, rv], [i, i], color=BLUE[200], lw=2, zorder=1)
        ax.scatter(dv, i, s=34, color=BLUE[250], edgecolor=SURFACE, linewidth=1.6, zorder=2)
        ax.scatter(rv, i, s=34, color=BLUE[600], edgecolor=SURFACE, linewidth=1.6, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_short(s) for s in reversed(order)])
    ax.set_xlim(0, 1)
    _pct(ax.xaxis)
    ax.set_xlabel("HAS-BLED final-score accuracy (T2 ∪ T3)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE[250], label="decision schema (D)"),
                       plt.Line2D([], [], marker="o", ls="", color=BLUE[600], label="raw-facts schema (R)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    page.figure(fig, "schema_variant", "Decision vs raw-facts schema (HAS-BLED)",
                "RQ3/H4. Same notes, two schemas: the extractor applies the thresholds (D) or the API does (R).")


def fig_cost(page: Page, cases: pd.DataFrame, systems: pd.DataFrame):
    plt = _plt()
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    g = d.groupby("system").agg(acc=("score_correct", "mean"), latency=("latency_s", "median")).reset_index()
    g = g.merge(systems[["system", "deployment", "family"]], on="system", how="left")
    deployments = ["on-prem CPU", "on-prem GPU", "external API"]
    colors = dict(zip(deployments, CATEGORICAL[:3]))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for dep in deployments:
        sub = g[g.deployment == dep]
        ax.scatter(sub.latency, sub.acc, s=40, color=colors[dep], edgecolor=SURFACE, linewidth=1.6, label=dep, zorder=3)
    best = g.sort_values("acc").groupby("family").tail(1)
    # fixed offsets per family keep the labels of neighbouring points apart
    offsets = {"Rules": (6, 4, "left"), "Encoder": (6, 4, "left"), "NuExtract": (-6, 8, "right"),
               "Open LLM": (-6, -12, "right"), "API LLM": (6, 6, "left")}
    for _, r in best.iterrows():
        dx, dy, ha = offsets.get(r.family, (6, 4, "left"))
        ax.annotate(_short(r.system), (r.latency, r.acc), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=INK_2, ha=ha)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    _pct(ax.yaxis)
    ax.set_xlabel("Median seconds per case (log scale, own hardware)")
    ax.set_ylabel("Final-score accuracy (T2 ∪ T3)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    page.figure(fig, "cost", "Accuracy against speed and deployment",
                "RQ5. Colour is where the system can run; this matters for patient data. The best system of each "
                "family is labelled. {tab:ops} adds GPU-hours.")


def fig_paper_arms(page: Page, eval_dir: Path):
    plt = _plt()
    import numpy as np

    path = eval_dir / "paper_arms.tsv"
    if not path.exists():
        return None
    arms = pd.read_csv(path, sep="\t")
    arms["acc"] = arms.correct / arms.n
    order = list(dict.fromkeys(arms.arm))
    models = list(dict.fromkeys(arms.model))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.3))
    x = np.arange(len(order))
    width = 0.32
    for i, m in enumerate(models):
        sub = arms[arms.model == m].set_index("arm").reindex(order)
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), sub.acc, width=width, color=CATEGORICAL[i], label=m, zorder=2)
        for xi, (arm, v) in zip(x, sub.acc.items()):
            if arm in ("omc (agent)", "extract→compute"):
                ax.text(xi + (i - 0.5) * (width + 0.02), v + 0.015, f"{100 * v:.0f}%", ha="center", fontsize=7.5,
                        color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.05)
    _pct(ax.yaxis)
    ax.set_ylabel("Final-score accuracy (T1)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    page.figure(fig, "paper_arms", "Extract-then-compute vs the paper's end-to-end arms",
                "RQ2/H3. The first five arms are the original study's (re-run on the same T1 cases); the last is "
                "this benchmark's pipeline with the same model. Only T1 allows the original arms' parser to be reused.")
    return arms


def fig_errors(page: Page, eval_dir: Path):
    plt = _plt()
    path = eval_dir / "error_codes.tsv"
    if not path.exists():
        return None
    e = pd.read_csv(path, sep="\t")
    canonical = ["MISS", "HALLUC", "NEG", "TEMP/UNIT", "THRESH", "ORD/LAT", "SCHEMA"]
    codes = [c for c in canonical if c in set(e.code)] + [c for c in dict.fromkeys(e.code) if c not in canonical]
    share = e.pivot_table(index="family", columns="code", values="n", aggfunc="sum").reindex(
        index=FAMILIES, columns=codes).fillna(0)
    share = share.div(share.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.45 * len(FAMILIES) + 0.6))
    left = pd.Series(0.0, index=share.index)
    for j, code in enumerate(codes):
        vals = share[code]
        ax.barh(range(len(FAMILIES)), vals, left=left, height=0.55, color=CATEGORICAL[j], label=code,
                edgecolor=SURFACE, linewidth=1.5, zorder=2)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v >= 0.09:
                ax.text(l0 + v / 2, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7.5,
                        color=INK if CATEGORICAL[j] == "#eda100" else "#ffffff")
        left += vals
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels(FAMILIES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(False)
    ax.legend(ncol=7, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5, handlelength=1.2)
    fig.tight_layout()
    page.figure(fig, "error_taxonomy", "What kind of mistakes each family makes",
                "Plan §10. Share of annotated field errors by code: missed fact, hallucinated value, negation, "
                "time point or unit, threshold applied wrongly, ordinal or laterality slip, schema failure. "
                "Real values need the two-annotator error study.")
    return share


def _confidence_rows(fields: pd.DataFrame) -> pd.DataFrame:
    if "confidence" not in fields:
        raise ValueError("no system reports per-field confidence")
    f = fields[(fields.variant == "D") & fields.confidence.notna()]
    prim = f[f.test_set.isin(PRIMARY)]
    f = prim if not prim.empty else f
    if f.empty:
        raise ValueError("no system reports per-field confidence")
    return f


def fig_selective(page: Page, fields: pd.DataFrame):
    """Risk-coverage: accuracy of the fields a system keeps when it defers the least confident."""
    plt = _plt()
    import numpy as np

    f = _confidence_rows(fields)
    systems = sorted(f.system.unique(), reverse=True)  # newest version first
    lows = []
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    for i, s in enumerate(systems):
        sub = f[f.system == s].sort_values("confidence", ascending=False)
        correct = sub.correct.astype(float).values
        coverage = np.arange(1, len(correct) + 1) / len(correct)
        accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        color = FAMILY_COLOR["System One"] if i == 0 else MUTED
        keep = coverage >= 0.02  # the first few fields are too few to estimate accuracy
        ax.plot(coverage[keep], accuracy[keep], color=color, lw=2, label=_short(s), solid_capstyle="round")
        lows.append(accuracy[keep].min())
    ax.set_xlim(0, 1)
    ax.set_ylim(max(0.0, min(lows) - 0.05), 1.0)
    ax.set_xlabel("Share of fields answered automatically (coverage)")
    ax.set_ylabel("Accuracy of those fields")
    _pct(ax.xaxis)
    _pct(ax.yaxis)
    ax.legend(loc="lower left")
    latest = f[f.system == systems[0]]
    prob_col = "probability" if "probability" in latest and latest.probability.notna().any() else "confidence"
    cal = stats.calibration(latest[prob_col].values, latest.correct.values, _clusters(latest).tolist())
    rel = pd.DataFrame(cal["reliability"])
    ax2.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", zorder=1)
    ax2.plot(rel.mean_confidence, rel.accuracy, color=FAMILY_COLOR["System One"], lw=1.6, zorder=2)
    ax2.scatter(rel.mean_confidence, rel.accuracy, s=8 + 60 * rel.n / rel.n.max(), color=FAMILY_COLOR["System One"],
                edgecolor=SURFACE, linewidth=1, zorder=3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    _pct(ax2.xaxis)
    _pct(ax2.yaxis)
    ax2.set_xlabel(f"Stated probability of the answer ({_short(systems[0])})")
    ax2.set_ylabel("Observed accuracy")
    ax2.text(0.04, 0.95, f"ECE {cal['ece']:.3f}\nBrier {cal['brier']:.3f}", transform=ax2.transAxes, va="top",
             fontsize=7.5, color=INK_2)
    _panel_letters([ax, ax2])
    fig.tight_layout()
    rel.insert(0, "system", systems[0])
    page.figure(fig, "confidence", "Using reported confidence to decide what to automate",
                "Systems that return a confidence per answer (Jev) can defer uncertain fields to a human. (a) "
                "accuracy of the fields kept as the least confident are deferred (risk-coverage; ranked by Jev's "
                "confidence statistic). (b) Reliability diagram of the probability Jev gave its chosen answer "
                "(10 equal-width bins, marker area ∝ fields; the dashed line is perfect calibration). "
                "{tab:confidence} gives operating points and {tab:calibration} the calibration metrics.",
                data=rel)


def table_calibration(page: Page, fields: pd.DataFrame):
    f = _confidence_rows(fields)
    rows = []
    for s in sorted(f.system.unique(), reverse=True):
        for scope, sub in [("all fields", f[f.system == s])] + [
                (k, g) for k, g in f[f.system == s].groupby("kind")]:
            prob_col = "probability" if "probability" in sub and sub.probability.notna().any() else "confidence"
            c = stats.calibration(sub[prob_col].values, sub.correct.values, _clusters(sub).tolist())
            rows.append({"system": _short(s), "fields": scope, "n": c["n"], "accuracy": c["accuracy"],
                         "mean stated probability": c["mean_confidence"],
                         "Brier (95% CI)": _est_ci({"n": 1, "est": c["brier"], "lo": c["brier_lo"], "hi": c["brier_hi"]}),
                         "ECE (95% CI)": _est_ci({"n": 1, "est": c["ece"], "lo": c["ece_lo"], "hi": c["ece_hi"]}),
                         "AUROC": c["auroc"], "AURC": c["aurc"]})
    page.table(pd.DataFrame(rows), "calibration", "Calibration and discrimination of reported confidence",
               "Plan §8.5 (H5). Brier score and expected calibration error (ECE, 10 equal-width bins) of the "
               "probability the system gave its chosen answer, with cluster-bootstrap CIs; AUROC = how well "
               "confidence separates right from wrong answers; AURC = area under the risk-coverage curve (lower is "
               "better). Mean stated probability above accuracy means overconfidence.")


def table_confidence(page: Page, fields: pd.DataFrame, cases: pd.DataFrame):
    f = _confidence_rows(fields)
    rows = []
    for s in sorted(f.system.unique(), reverse=True):
        sub = f[f.system == s]
        per_case = sub.groupby("id").confidence.min()
        case_ok = cases[(cases.system == s) & (cases.variant == "D")].set_index("id").score_correct
        for tau in (0.5, 0.7, 0.8, 0.9):
            kept = sub[sub.confidence >= tau]
            auto_cases = per_case[per_case >= tau].index
            rows.append({"system": _short(s), "threshold": tau,
                         "fields auto-accepted": len(kept) / len(sub),
                         "accuracy of auto-accepted fields": kept.correct.mean() if len(kept) else float("nan"),
                         "cases fully automatic": len(auto_cases) / len(per_case),
                         "score accuracy of those cases": case_ok.reindex(auto_cases).mean() if len(auto_cases)
                         else float("nan")})
    page.table(pd.DataFrame(rows), "confidence", "Confidence-gated operating points",
               "A case is fully automatic when every field's confidence clears the threshold; the rest go to "
               "review. The clinical question is which threshold gives acceptable score accuracy at useful coverage.")


def fig_attribution_real(page: Page):
    plt = _plt()
    path = ds.DATA_DIR / "a409_attribution.jsonl"
    if not path.exists():
        return
    rows = pd.DataFrame(list(ds.read_jsonl(path)))
    rows = rows[rows.k.notna()]
    g = rows.groupby("calculator").agg(observed=("k", lambda k: (k == 1).mean()),
                                       mirrored=("mirrored_k1", "mean"), n=("k", "size"))
    g = g.sort_values("observed")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (calc, r) in enumerate(g.iterrows()):
        ax.plot([r.mirrored, r.observed], [i, i], color=BLUE[200], lw=2, zorder=1)
        ax.scatter(r.mirrored, i, s=34, color=BLUE[250], edgecolor=SURFACE, linewidth=1.6, zorder=2)
        ax.scatter(r.observed, i, s=34, color=BLUE[600], edgecolor=SURFACE, linewidth=1.6, zorder=3)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{c} (n={int(n)})" for c, n in zip(g.index, g.n)])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_xlabel("Share of failures reproduced by changing one field")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE[600], label="observed error"),
                       plt.Line2D([], [], marker="o", ls="", color=BLUE[250], label="mirrored error (control)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    page.figure(fig, "a409_attribution", "Can a final-score error be traced to one field?",
                "Real data: the 373 Abacus-409 failures with a numeric answer. Where the control is as high as the "
                "observed share (additive scores), a one-field explanation is no evidence of which field was misread.",
                real=True)


def fig_baselines_real(page: Page):
    plt = _plt()
    path = ds.DATA_DIR / "results_cpu_baselines" / "summary.tsv"
    if not path.exists():
        return
    s = pd.read_csv(path, sep="\t")
    s = s[(s.calculator == "ALL") & (s.variant == "D") & (s.system != "oracle")]
    systems = ["majority", "template_inversion", "rules"]
    labels = {"majority": "majority class (floor)", "template_inversion": "template inversion (memorisation)",
              "rules": "rules (written on T1/T4)"}
    sets = [x for x in ["T1", "A409", "T4", "T5"] if x in set(s.test_set)]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    offsets = {"majority": -0.18, "template_inversion": 0.0, "rules": 0.18}
    for i, sys_ in enumerate(systems):
        sub = s[s.system == sys_].set_index("test_set").reindex(sets)
        ax.scatter([j + offsets[sys_] for j in range(len(sets))], sub.score_acc, s=40, color=CATEGORICAL[i],
                   edgecolor=SURFACE, linewidth=1.6, label=labels[sys_], zorder=3)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([SET_LABEL[x] for x in sets])
    ax.set_ylim(0, 1.05)
    _pct(ax.yaxis)
    ax.set_ylabel("Final-score accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.5)
    fig.tight_layout()
    page.figure(fig, "cpu_baselines", "CPU baselines across test sets",
                "Real data. Perfect scores on template text (T1, A409) fall to the floor once the wording is new: "
                "the reason T2, T3 and T5 are primary.", real=True)


# ---------------------------------------------------------------- tables


def _yaml(path: Path) -> Dict:
    import yaml

    return yaml.safe_load(path.read_text()) if path.exists() else {}


def fig_flow(page: Page):
    """Where every test case comes from, with the counts in the repository (CONSORT/STARD-style flow)."""
    from collections import Counter

    from matplotlib.patches import FancyBboxPatch

    plt = _plt()
    sets = {n: list(ds.read_jsonl(ds.DATA_DIR / f"{n}.jsonl")) for n in ("T1", "A409", "T4", "T5")
            if (ds.DATA_DIR / f"{n}.jsonl").exists()}
    a409 = _yaml(ds.DATA_DIR / "A409_match_report.yaml")
    t5 = _yaml(ds.DATA_DIR / "T5_build_report.yaml")
    n = {k: len(v) for k, v in sets.items()}
    t4_kinds = Counter(r["meta"]["perturbation"] for r in sets.get("T4", []))
    t4_bases = len({r["meta"]["base_id"] for r in sets.get("T4", [])})
    t1_calcs = len({r["calculator"] for r in sets.get("T1", [])})

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(12, 67)
    ax.axis("off")

    def box(x, y, w, h, text, kind="source"):
        fill = {"source": "#ffffff", "set": BLUE[100], "excluded": NEUTRAL_MID, "pending": SURFACE}[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6", fc=fill,
                                    ec=BLUE[400] if kind == "set" else AXIS, lw=0.8,
                                    ls="--" if kind == "pending" else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.9, color=INK, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                                            shrinkA=0, shrinkB=0, mutation_scale=7))

    top, mid, low, h = 56, 32, 14, 10  # row baselines; CONSORT layout: exclusions branch off to the right
    # generator: T1, its perturbations (T4), the training pool and the new-wording sets
    box(1, top, 20, h, "Original vignette generator\n(seed-exact copy)\n10 calculators")
    box(1, mid, 20, h, f"T1 template vignettes\nn = {n.get('T1', 0):,}\n({t1_calcs} calculators × 100)", "set")
    box(1, low, 20, h, f"T4 perturbed copies\nn = {n.get('T4', 0):,}\n{len(t4_kinds)} perturbation types\n"
        f"of {t4_bases} T1 cases", "set")
    box(25, top, 20, h, "TRAIN pool\ndisjoint phrasings; any\ntest vignette removed")
    box(25, mid, 20, h, "T2 new phrasings\nT3 whole-note paraphrases\n≈1,000 each, built after\nclinician review",
        "pending")
    arrow(11, top, 11, mid + h)
    arrow(11, mid, 11, low + h)
    arrow(21, top + h / 2, 25, top + h / 2)
    arrow(21, mid + h / 2, 25, mid + h / 2)
    # Abacus-409: the paper's released failures
    box(49, top, 18, h, f"Abacus-409\nreleased failures\nn = {a409.get('rows', 0)}")
    box(49, mid, 18, h, f"A409 matched to seeds\nn = {n.get('A409', 0)}", "set")
    box(59.5, 44, 13.5, 7.5, f"Excluded\nempty vignette: {a409.get('empty_vignette', 0)}\n"
        f"unmatched: {a409.get('unmatched', 0)}", "excluded")
    arrow(58, top, 58, mid + h)
    arrow(58, 47.75, 59.5, 47.75)
    # MedCalc-Bench Verified: real case-report notes
    box(76, top, 23, h, f"MedCalc-Bench Verified\n(pinned revision)\n{t5.get('rows', 0)} rows, 7 shared "
        "calculators")
    box(76, mid, 23, h, f"T5 case-report notes\nn = {n.get('T5', 0)}\nOMC = MedCalc-Bench: "
        f"{t5.get('score_agrees', 0)}", "set")
    box(88.5, 44, 11, 7.5, f"Excluded\noutside OMC\nranges: {t5.get('invalid_request', 0)}", "excluded")
    arrow(87.5, top, 87.5, mid + h)
    arrow(87.5, 47.75, 88.5, 47.75)
    ax.text(49, 22, "Blue: evaluated test sets.  Grey: exclusions.\nDashed: built after clinician review.",
            fontsize=6.2, color=MUTED, va="center", linespacing=1.4)
    flows = pd.DataFrame([
        {"stage": "T1", "n": n.get("T1", 0)}, {"stage": "T4", "n": n.get("T4", 0)},
        {"stage": "T4 base cases", "n": t4_bases},
        *({"stage": f"T4 {k}", "n": v} for k, v in sorted(t4_kinds.items())),
        {"stage": "Abacus-409 rows", "n": a409.get("rows", 0)},
        {"stage": "Abacus-409 empty vignette", "n": a409.get("empty_vignette", 0)},
        {"stage": "Abacus-409 unmatched", "n": a409.get("unmatched", 0)}, {"stage": "A409", "n": n.get("A409", 0)},
        {"stage": "MedCalc-Bench rows", "n": t5.get("rows", 0)},
        {"stage": "MedCalc-Bench invalid request", "n": t5.get("invalid_request", 0)},
        {"stage": "T5", "n": n.get("T5", 0)}, {"stage": "T5 OMC agrees", "n": t5.get("score_agrees", 0)}])
    page.figure(fig, "flow", "Where the test cases come from",
                "Counts are read from the committed datasets and build reports. T5 is rebuilt locally from "
                "MedCalc-Bench Verified (CC-BY-SA) and not redistributed; `MANIFEST.tsv` holds the SHA-256 of every "
                "set so a rebuild can be checked byte for byte.", real=True, data=flows)


def table_datasets(page: Page):
    rows = []
    for name in ["T1", "A409", "T4", "T5"]:
        path = ds.DATA_DIR / f"{name}.jsonl"
        if not path.exists():
            continue
        recs = list(ds.read_jsonl(path))
        n_fields = sum(len(r["gold"]["D"]) for r in recs)
        undocumented = sum(v == "not documented" for r in recs for v in r["field_sources"]["D"].values())
        rows.append({"set": name, "cases": len(recs), "calculators": len({r["calculator"] for r in recs}),
                     "field instances": n_fields, "undocumented": f"{100 * undocumented / n_fields:.1f}%",
                     "HAS-BLED raw-facts (R)": sum("R" in r["gold"] for r in recs)})
    rows += [{"set": "T2", "cases": "≈1,000 (planned)", "calculators": 10, "field instances": "—",
              "undocumented": "—", "HAS-BLED raw-facts (R)": "≈100"},
             {"set": "T3", "cases": "≈1,000 (planned)", "calculators": 10, "field instances": "—",
              "undocumented": "—", "HAS-BLED raw-facts (R)": "≈100"}]
    page.table(pd.DataFrame(rows), "datasets", "Test sets",
               "T5 is built locally from MedCalc-Bench Verified (CC-BY-SA) and is not committed. "
               "T2/T3 sizes are targets until clinician review is complete.", real=True)


def table_power(page: Page):
    """Design-based precision and detectable differences (does not depend on results)."""
    rows = []
    for label, n in [("T2 ∪ T3 (primary)", 2000), ("one test set", 1000), ("T5 (MedCalc-Bench, 7 calculators)", 637),
                     ("one calculator in T2 ∪ T3", 200)]:
        row = {"comparison set": label, "cases": n, "±95% CI at 90% accuracy": stats.ci_half_width(0.9, n)}
        for psi in (0.05, 0.10, 0.20):
            row[f"MDE, {psi:.0%} discordant"] = stats.mde_paired(n, psi)
        row["MDE, 10% discordant, Holm worst case (α=0.05/15)"] = stats.mde_paired(n, 0.10, alpha=0.05 / 15)
        rows.append(row)
    page.table(pd.DataFrame(rows), "power", "Precision and minimum detectable differences",
               "Plan §9. CI half-width for an accuracy of 90% assuming independent cases (clustering widens it; "
               "T2 and T3 cases are independent of each other). MDE = smallest paired difference in final-score "
               "accuracy detected with 80% power by the two-sided McNemar test (Connor 1987), for a given share of "
               "cases on which two systems disagree; the last column applies the most stringent Holm step for the "
               "15 between-family comparisons. Per-calculator results are descriptive.", real=True)


def table_systems(page: Page, systems: pd.DataFrame):
    manifest = Path(__file__).resolve().parents[2] / "cluster" / "killarney" / "models.tsv"
    gpus = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 5 and not line.startswith("#"):
                gpus.setdefault(parts[1].split("/")[-1], f"{parts[3]} × {parts[2].upper()}")
    rows = []
    for _, r in systems.iterrows():
        key = _short(r.system).split(" (")[0]
        hw = next((v for k, v in gpus.items() if k.replace("-Instruct", "") == key or key in k), "")
        if r.family == "Rules":
            hw = "CPU"
        if r.family == "API LLM":
            hw = "external API"
        rows.append({"system": _short(r.system), "family": r.family, "deployment": r.deployment,
                     "Killarney resources": hw})
    page.table(pd.DataFrame(rows), "systems", "Systems under comparison",
               "From `cluster/killarney/models.tsv`. Encoders are fine-tuned on TRAIN; all LLM rows use "
               "JSON-schema constrained decoding unless marked otherwise.", real=True)


def table_primary(page: Page, cases: pd.DataFrame, fields: pd.DataFrame):
    d = cases[cases.variant == "D"]
    f = fields[fields.variant == "D"]
    rows = []
    for s in _system_order(cases):
        ds_ = d[d.system == s]
        prim = ds_[ds_.test_set.isin(PRIMARY)]
        row = {"system": _short(s), "family": ds_.family.iloc[0], "n (T2∪T3)": len(prim),
               "T2∪T3 score": _est_ci(_boot(prim)),
               "T2∪T3 macro field": _est_ci(_macro_boot(f[(f.system == s) & f.test_set.isin(PRIMARY)]))}
        for t in ("T1", "T4", "T5"):
            sub = ds_[ds_.test_set == t]
            row[f"{t} score"] = _est_ci(_boot(sub)) if len(sub) else "—"
        row["valid"] = prim.valid.mean()
        rows.append(row)
    page.table(pd.DataFrame(rows), "primary", "Primary endpoints",
               "Estimate (95% CI). Score = final-score accuracy, OpenMedCalc on the extracted request. Macro field = "
               "per-field accuracy averaged within calculator, then across calculators. CIs are percentile intervals "
               "from 2,000 cluster-bootstrap resamples of base cases. Valid = schema-valid outputs.")


def _best_per_family(cases: pd.DataFrame) -> Dict[str, str]:
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    acc = d.groupby(["family", "system"]).score_correct.mean().reset_index()
    return {f: g.sort_values("score_correct").system.iloc[-1] for f, g in acc.groupby("family")}


def pairwise_family(cases: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified comparison family (i): best system of each family against each other, T2 ∪ T3."""
    best = _best_per_family(cases)
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    f = fields[(fields.variant == "D") & fields.test_set.isin(PRIMARY)]
    fams = [x for x in FAMILIES if x in best]
    rows = []
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            r = compare(d, f, best[a], best[b])
            rows.append({"comparison": f"{a} vs {b}", "first": best[a], "second": best[b], "n": r["n_cases"],
                         "only first right": r["score"]["a_only"], "only second right": r["score"]["b_only"],
                         "d_score": r["score"]["est"], "d_score_lo": r["score"]["lo"], "d_score_hi": r["score"]["hi"],
                         "p_mcnemar": r["score"]["p"], "d_macro": r["macro"]["est"], "d_macro_lo": r["macro"]["lo"],
                         "d_macro_hi": r["macro"]["hi"]})
    out = pd.DataFrame(rows)
    adj = stats.holm(dict(zip(out.comparison, out.p_mcnemar)))
    out["p_holm"] = out.comparison.map(adj)
    return out


def _pairwise_data(page: Page, cases: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    return pairwise_family(cases, fields)


def table_pairwise(page: Page, comparisons: pd.DataFrame):
    t = pd.DataFrame({
        "comparison": comparisons.comparison,
        "systems": [f"{_short(a)} vs {_short(b)}" for a, b in zip(comparisons["first"], comparisons["second"])],
        "n": comparisons.n, "only first right": comparisons["only first right"],
        "only second right": comparisons["only second right"],
        "Δ score accuracy (95% CI)": [_est_ci({"n": 1, "est": e, "lo": lo, "hi": hi}) for e, lo, hi in
                                      zip(comparisons.d_score, comparisons.d_score_lo, comparisons.d_score_hi)],
        "Δ macro field (95% CI)": [_est_ci({"n": 1, "est": e, "lo": lo, "hi": hi}) for e, lo, hi in
                                   zip(comparisons.d_macro, comparisons.d_macro_lo, comparisons.d_macro_hi)],
        "p (exact McNemar)": comparisons.p_mcnemar.map(_p), "p (Holm)": comparisons.p_holm.map(_p)})
    page.table(t, "pairwise", "Pairwise comparisons of the best system per family",
               "Comparison family (i) of plan §9, on T2 ∪ T3, paired by case. Δ = first − second. Score-accuracy "
               "differences are tested with the exact McNemar test, Holm-adjusted within the family; CIs for both "
               "differences come from the paired cluster bootstrap. {fig:forest} plots the differences.")


def fig_forest(page: Page, comparisons: pd.DataFrame):
    plt = _plt()
    n = len(comparisons)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 0.26 * n + 1.2), sharey=True)
    ys = list(range(n))[::-1]
    for ax, (col, label) in zip(axes, [("d_score", "Δ final-score accuracy"), ("d_macro", "Δ macro field accuracy")]):
        ax.axvline(0, color=AXIS, lw=1, zorder=1)
        for y_, (_, r) in zip(ys, comparisons.iterrows()):
            # score: Holm-adjusted McNemar; macro field (no test): CI excludes zero
            sig = r.p_holm < 0.05 if col == "d_score" else (r[f"{col}_lo"] > 0 or r[f"{col}_hi"] < 0)
            ax.plot([r[f"{col}_lo"], r[f"{col}_hi"]], [y_, y_], color=INK_2 if sig else AXIS, lw=1.6,
                    solid_capstyle="round", zorder=2)
            ax.scatter(r[col], y_, s=26, color=INK if sig else MUTED, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        lim = max(0.05, comparisons[[f"{col}_lo", f"{col}_hi"]].abs().max().max() * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(label, fontsize=8.5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:+.0f} pp" if v else "0"))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(comparisons.comparison)
    _panel_letters(axes)
    fig.tight_layout()
    page.figure(fig, "forest", "Paired differences between the best system of each family (T2 ∪ T3)",
                "Points right of zero favour the first-named family. 95% paired cluster-bootstrap CIs. Dark marks: "
                "(a) score-accuracy differences significant after Holm adjustment; (b) macro field-accuracy "
                "differences whose CI excludes zero. Table view: {tab:pairwise}.",
                data=comparisons)


def table_variant_paired(page: Page, cases: pd.DataFrame, fields: pd.DataFrame):
    """Comparison family (ii): raw-facts (R) vs decision (D) schema within each system, HAS-BLED, T2 ∪ T3."""
    d = cases[(cases.calculator == "hasbled") & cases.test_set.isin(PRIMARY)]
    systems = [s for s in _system_order(cases) if {"D", "R"} <= set(d[d.system == s].variant)]
    if not systems:
        raise ValueError("no system was run with both schema variants")
    rows = []
    for s in systems:
        sub = d[d.system == s]
        wide = sub.pivot_table(index=["id", "cluster"], columns="variant", values="score_correct",
                               aggfunc="first").dropna().reset_index()
        r = stats.paired(wide["R"].astype(bool).values, wide["D"].astype(bool).values, wide.cluster.tolist())
        rows.append({"system": _short(s), "n": len(wide), "D": wide["D"].mean(), "R": wide["R"].mean(),
                     "Δ R − D (95% CI)": _est_ci(r), "p": r["p"]})
    adj = stats.holm({row["system"]: row["p"] for row in rows})
    for row in rows:
        row["p (Holm)"] = _p(adj[row["system"]])
        row["p"] = _p(row["p"])
    page.table(pd.DataFrame(rows), "variant_paired", "Raw-facts vs decision schema within each system (HAS-BLED)",
               "Comparison family (ii) of plan §9 (H4): final-score accuracy on the same T2 ∪ T3 HAS-BLED notes with "
               "each schema. Exact McNemar test, Holm-adjusted across systems; paired cluster-bootstrap CI.")


def table_clinician(page: Page, cases: pd.DataFrame, clin: Dict):
    """Comparison family (iv): best system of each family against the clinicians, on the clinicians' sample."""
    cc = clin["cases"]
    if cc.empty:
        raise ValueError("no clinician predictions yet (python -m llm_calc.extraction clinician)")
    ref = cc.groupby(["id", "cluster"]).score_correct.mean().rename("clinicians").reset_index()
    readers = sorted(cc.system.unique())
    per_reader = cc.pivot_table(index="id", columns="system", values="score_correct", aggfunc="first")
    best = _best_per_family(cases)
    d = cases[cases.variant == "D"]
    rows = []
    for fam in [f for f in FAMILIES if f in best]:
        s = best[fam]
        m = ref.merge(d[d.system == s][["id", "score_correct"]], on="id")
        r = stats.cluster_bootstrap(m.score_correct.astype(float) - m.clinicians, m.cluster.tolist())
        row = {"family": fam, "system": _short(s), "n": len(m), "system accuracy": m.score_correct.mean(),
               "clinicians (mean of readers)": m.clinicians.mean(),
               "Δ system − clinicians (95% CI)": _est_ci(r)}
        for reader in readers:
            pair = per_reader[reader].reindex(m.id).values
            ok = ~pd.isna(pair)
            row[f"p vs {reader.split(':')[-1]} (McNemar)"] = stats.mcnemar_exact(
                m.score_correct.values[ok].astype(bool), pair[ok].astype(bool))["p"]
        rows.append(row)
    t = pd.DataFrame(rows)
    pcols = [c for c in t if c.startswith("p vs")]
    flat = {f"{i}|{c}": t.loc[i, c] for i in t.index for c in pcols}
    adj = stats.holm(flat)
    for c in pcols:
        t[c] = [_p(adj[f"{i}|{c}"]) for i in t.index]
    both = per_reader.dropna()
    agree = (both[readers[0]] == both[readers[-1]]).mean() if len(readers) > 1 else float("nan")
    secs = cc.groupby("system").latency_s.median()
    page.table(t, "clinician", "Systems against the clinician reference",
               f"Comparison family (iv): each family's best system against two clinicians who filled the same typed "
               f"request blind to gold ({len(ref)} cases, stratified by set and calculator; `clinician.py`). Δ uses "
               f"the mean of the two readers per case, with a cluster-bootstrap CI; McNemar tests are against each "
               f"reader, Holm-adjusted across the table. The readers agreed on the final score for {agree:.1%} of "
               f"cases; median time per case "
               + ", ".join(f"{r.split(':')[-1]} {secs[r]:.0f} s" for r in readers) + ".")


def table_sensitivity(page: Page, cases: pd.DataFrame, fields: pd.DataFrame):
    """Do the primary conclusions survive analysis choices made before seeing results?"""
    best = _best_per_family(cases)
    d = cases[cases.variant == "D"]
    f = fields[fields.variant == "D"]
    open_decisions = {"hasbled", "cci"}  # plan §13: HAS-BLED age rule, CCI liver wording
    rows = []
    for fam in [x for x in FAMILIES if x in best]:
        s = best[fam]
        prim = d[(d.system == s) & d.test_set.isin(PRIMARY)]
        fp = f[(f.system == s) & f.test_set.isin(PRIMARY)]
        t5 = d[(d.system == s) & (d.test_set == "T5")]
        row = {"family": fam, "system": _short(s),
               "S0 primary (score, T2∪T3)": _boot(prim)["est"],
               "S1 invalid outputs excluded": _boot(prim[prim.valid])["est"],
               "S2 without HAS-BLED and CCI": _boot(prim[~prim.calculator.isin(open_decisions)])["est"],
               "S3 macro field, exact floats": float("nan"),
               "S4 T5, all rows": _boot(t5)["est"] if len(t5) else float("nan"),
               "S5 T5, OMC agrees with MedCalc-Bench": float("nan")}
        if "correct_exact" in fp and fp.correct_exact.notna().any():
            row["S3 macro field, exact floats"] = _macro_boot(fp.assign(correct=fp.correct_exact.astype(float)))["est"]
        if len(t5) and "external_agrees" in t5:
            row["S5 T5, OMC agrees with MedCalc-Bench"] = _boot(t5[t5.external_agrees == True])["est"]  # noqa: E712
        rows.append(row)
    t = pd.DataFrame(rows)
    rank = t.set_index("family").drop(columns="system").rank(ascending=False).astype("Int64")
    t["family rank stable"] = ["yes" if rank.loc[fam].dropna().nunique() == 1 else "no" for fam in t.family]
    page.table(t, "sensitivity", "Sensitivity analyses",
               "Pre-specified (plan §9.1). S0: primary endpoint. S1: invalid outputs dropped instead of counted wrong "
               "(favours systems that fail loudly). S2: calculators whose scoring has an open decision removed "
               "(HAS-BLED age ≥65 vs >65; CCI liver wording). S3: macro field accuracy with exact float match instead "
               "of ±1%. S4–S5: T5 with and without the rows where OpenMedCalc's score differs from MedCalc-Bench's "
               "answer. The last column says whether each family keeps its rank across all of them.")


def table_subgroups(page: Page, cases: pd.DataFrame):
    best = _best_per_family(cases)
    d = cases[(cases.variant == "D") & cases.system.isin(best.values())]
    groups = []
    prim = d[d.test_set.isin(PRIMARY)]
    kind = {"meldna": "lab equation", "sofa": "organ physiology", "nihss": "examination scale",
            "gad7": "questionnaire"}
    groups += [("calculator type (T2∪T3)", kind.get(c, "criteria checklist"), sub)
               for c, sub in prim.groupby("calculator")]
    t5 = d[d.test_set == "T5"]
    if len(t5) and "note_type" in t5 and t5.note_type.notna().any():
        groups += [("T5 note type", str(nt), sub) for nt, sub in t5.groupby("note_type")]
    if "perturbation" in d:
        t4 = d[d.test_set == "T4"]
        groups += [("T4 perturbation", p_, sub) for p_, sub in t4.groupby("perturbation")]
    rows = []
    merged: Dict[tuple, List[pd.DataFrame]] = {}
    for dim, level, sub in groups:
        merged.setdefault((dim, level), []).append(sub)
    for (dim, level), parts in merged.items():
        sub = pd.concat(parts)
        row = {"subgroup": dim, "level": level, "n cases": sub.id.nunique()}
        for fam in [x for x in FAMILIES if x in best]:
            row[fam] = _est_ci(_boot(sub[sub.system == best[fam]]), 2)
        rows.append(row)
    page.table(pd.DataFrame(rows), "subgroups", "Final-score accuracy by subgroup (best system per family)",
               "Descriptive (plan §9.2): subgroup n is small, so differences between subgroups are not tested. "
               "Calculator type groups calculators by the kind of input they need. T5 note type separates "
               "MedCalc-Bench's synthetic notes from notes extracted from published case reports.")


def table_per_calculator(page: Page, cases: pd.DataFrame):
    best = _best_per_family(cases)
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY) & cases.system.isin(best.values())]
    acc = d.groupby(["calculator", "system"]).score_correct.mean().unstack()
    acc = acc[[best[f] for f in FAMILIES if f in best]]
    acc.columns = [f"{f}: {_short(best[f])}" for f in FAMILIES if f in best]
    acc.insert(0, "n", d.groupby("calculator").id.nunique())
    page.table(acc.reset_index(), "per_calculator", "Final-score accuracy by calculator (best system per family, T2 ∪ T3)",
               "Per-calculator n is about 200 per system, so these rows are descriptive (plan §9).")


def table_band(page: Page, cases: pd.DataFrame):
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    g = d.groupby("system").agg(score=("score_correct", "mean"), band=("band_correct", "mean"),
                                all_fields=("all_fields_correct", "mean"), mae=("abs_error", "mean"))
    g = g.reindex(_system_order(cases)).reset_index()
    g["system"] = g.system.map(_short)
    g.columns = ["system", "score accuracy", "risk-band agreement", "all fields correct", "mean |score error|"]
    page.table(g, "band", "Clinical agreement: risk bands, all-fields rate, score error",
               "Risk bands are provisional (`evaluate.RISK_BANDS`) until a clinician signs off the cut-offs. "
               "The gap between all-fields-correct and score accuracy measures how often errors cancel out.")


def table_ops(page: Page, cases: pd.DataFrame, systems: pd.DataFrame):
    d = cases[cases.variant == "D"]
    g = d.groupby("system").latency_s.agg(["median", lambda x: x.quantile(0.95)]).reset_index()
    g.columns = ["system", "median s/case", "p95 s/case"]
    g = g.merge(systems[["system", "deployment"]], on="system", how="left")
    g["GPU-hours per 1k cases"] = [(m * 1000 / 3600) if dep == "on-prem GPU" else float("nan")
                                   for m, dep in zip(g["median s/case"], g.deployment)]
    g = g.set_index("system").reindex(_system_order(cases)).reset_index()
    g["system"] = g.system.map(_short)
    page.table(g, "ops", "Operational metrics",
               "Plan §8.4. Real latency comes from `latency_s` in each prediction; GPU-hours scale it by the GPUs "
               "in `models.tsv`; API cost is added from the provider's invoice.")


def _term_label(term: str, reference: str, systems: List[str]) -> str:
    t = (term.replace(f"C(system, Treatment({reference!r}))", "system")
         .replace("C(test_set, Treatment('T1'))", "set").replace("C(kind, Treatment('bool'))", "field type")
         .replace("C(calculator)", "calculator").replace("[T.", ": ").replace("]", "").replace(":set", " × set"))
    for s in sorted(systems, key=len, reverse=True):
        t = t.replace(f"system: {s}", f"system: {_short(s)}")
    return t


def table_model(page: Page, fields: pd.DataFrame, cases: pd.DataFrame):
    best = _best_per_family(cases)
    f = fields[(fields.variant == "D") & fields.system.isin(best.values())
               & fields.test_set.isin(["T1", "T2", "T3", "T4", "T5"])]
    reference = best.get("Rules") or sorted(best.values())[0]
    res = stats.gee_field_model(f, reference_system=reference)
    coef = res["coefficients"]
    coef = coef[~coef.term.str.startswith("C(calculator)")]  # nuisance adjustment, not reported
    t = pd.DataFrame({"term": [_term_label(x, reference, list(best.values())) for x in coef.term],
                      "odds ratio (95% CI)": [f"{o:.2f} ({lo:.2f} to {hi:.2f})" for o, lo, hi in
                                              zip(coef["or"], coef.or_lo, coef.or_hi)],
                      "p": coef.p.map(_p)})
    page.table(t, "model", "Field-level model: which factors change the odds of a correct field",
               f"Population-averaged logistic model (GEE, robust sandwich SEs clustered by base case; "
               f"{res['n_rows']:,} field rows in {res['n_clusters']:,} clusters): correct ~ system × test set + field "
               f"type + calculator, best system per family, decision schema. Reference levels: "
               f"{_short(reference)}, T1, boolean field; calculator effects are adjusted for but not shown. "
               "Interaction rows (system × set) say how much more (OR > 1) or less a system degrades on that set than "
               "the reference does; their joint test is in {tab:model_terms}.")
    terms = res["terms"]
    terms = terms[terms.term != "Intercept"]
    exch = stats.gee_field_model(f, reference_system=reference, working="exchangeable")["terms"].set_index("term")
    page.table(pd.DataFrame({"term": [_term_label(x, reference, list(best.values())) for x in terms.term],
                             "χ²": terms.chi2.round(1), "df": terms["df"], "p": terms.p.map(_p),
                             "χ², exchangeable (S6)": exch.chi2.reindex(terms.term).round(1).values,
                             "p, exchangeable (S6)": exch.p.reindex(terms.term).map(_p).values}),
               "model_terms", "Joint Wald tests for the field-level model",
               "The system × test set row is the pre-specified test of H1–H2: do systems differ in how much they "
               "lose from template text (T1) to new wording (T2, T3), perturbations (T4) and real notes (T5)? The last two "
               "columns refit the model with an exchangeable working correlation (sensitivity S6).")


def _try(page: Page, fn, *args):
    """Render one figure or table; record why if the data for it does not exist yet."""
    try:
        return fn(page, *args)
    except Exception as e:  # noqa: BLE001 - a missing set must not stop the rest of the page
        name = fn.__name__.replace("_", " ")
        page.text(f"_{name}: not rendered ({type(e).__name__}: {str(e).splitlines()[0][:120]})._\n")
        return None


def write_all(eval_dir: Path, out: Path, mock: bool) -> Path:
    cases, fields, systems, clin = load(eval_dir)
    page = Page(out, mock)
    page.text("## Design\n\n```mermaid\nflowchart LR\n  N[Clinical text] --> X[Extractor<br/>rules · encoder · NuExtract · LLM]\n"
              "  X --> R[Typed request<br/>Pydantic, validated]\n  R --> C[OpenMedCalc]\n  C --> S[Score]\n"
              "  R -. per-field grading .-> G1[(Gold request)]\n  S -. score grading .-> G2[(Gold score)]\n```\n\n"
              "{figdef:design}. Every system fills the same typed request; the calculator is deterministic, so all "
              "errors are extraction errors.\n")
    page.text("## Data\n")
    _try(page, fig_flow)
    _try(page, table_datasets)
    _try(page, table_systems, systems)
    _try(page, table_power)
    page.text("## Primary endpoints\n")
    _try(page, fig_headline, cases, clin)
    _try(page, table_primary, cases, fields)
    _try(page, fig_generalisation, cases)
    comparisons = _try(page, _pairwise_data, cases, fields)
    if comparisons is not None:
        _try(page, table_pairwise, comparisons)
        _try(page, fig_forest, comparisons)
    _try(page, table_clinician, cases, clin)
    page.text("## Where systems succeed and fail\n")
    _try(page, fig_heatmap, fields, cases)
    _try(page, table_per_calculator, cases)
    _try(page, fig_field_kind, fields)
    _try(page, fig_assertion, fields)
    _try(page, table_band, cases)
    _try(page, table_subgroups, cases)
    _try(page, table_sensitivity, cases, fields)
    page.text("## Robustness, schema design and the original study\n")
    _try(page, fig_t4, cases)
    _try(page, fig_variant, cases)
    _try(page, table_variant_paired, cases, fields)
    arms = _try(page, fig_paper_arms, eval_dir)
    if arms is not None:
        t = arms.pivot_table(index="arm", columns="model", values="acc", sort=False).reset_index()
        page.table(t, "paper_arms", "Paper arms vs extract-then-compute (T1)", "Table view of {fig:paper_arms}.")
    page.text("## Errors, cost and the statistical model\n")
    share = _try(page, fig_errors, eval_dir)
    if share is not None:
        page.table(share.reset_index().rename(columns={"index": "family"}), "error_codes", "Error codes by family (share of annotated field errors)", "Table view of {fig:error_taxonomy}.")
    _try(page, fig_cost, cases, systems)
    _try(page, table_ops, cases, systems)
    _try(page, table_model, fields, cases)
    page.text("## Confidence and selective automation\n")
    _try(page, fig_selective, fields)
    _try(page, table_confidence, fields, cases)
    _try(page, table_calibration, fields)
    page.text("## Real results already available\n")
    _try(page, fig_attribution_real)
    _try(page, fig_baselines_real)
    header = ("# Extraction benchmark: figures and tables\n\n"
              + ("> **Mock shells.** Every figure and table marked MOCK is drawn from mock predictions "
                 "(`llm_calc/extraction/mock.py`): gold labels corrupted at *assumed* accuracies, then graded by the "
                 "real evaluator. They show what each experiment will produce and how it will be read. They are not "
                 "results and must not be cited. Items marked REAL use data that already exists.\n\n"
                 "Regenerate: `python -m llm_calc.extraction mock-results --out <dir>` then "
                 "`python -m llm_calc.extraction figures --eval-dir <dir> --out docs/mock_results --mock`. "
                 "The same `figures` command renders the real page from the Killarney evaluation directory "
                 "(without `--mock`).\n" if mock else
                 "Generated by `python -m llm_calc.extraction figures`.\n"))
    return page.write(header)
