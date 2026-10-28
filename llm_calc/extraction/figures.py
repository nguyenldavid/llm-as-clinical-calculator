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
PERTURBATIONS = ["distractors", "units", "conflicting", "abbreviations", "reorder", "long_note"]


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
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.parts: List[str] = []

    def label(self, real: bool, shell: bool = False) -> str:
        if shell:
            return "SHELL"
        return "REAL" if real else ("MOCK" if self.mock else "")

    def figure(self, fig, name: str, title: str, caption: str, real: bool = False):
        plt = _plt()
        if self.mock and not real:
            fig.text(0.5, 0.5, "MOCK DATA · ILLUSTRATIVE ONLY", ha="center", va="center", rotation=24,
                     fontsize=34, color=MUTED, alpha=0.16, weight="bold", zorder=100)
        fig.savefig(self.fig_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        tag = self.label(real)
        self.parts.append(f"### {title}{f' ({tag})' if tag else ''}\n\n![{title}](figures/{name}.png)\n\n{caption}\n")

    def table(self, df: pd.DataFrame, title: str, caption: str = "", real: bool = False, shell: bool = False):
        tag = self.label(real, shell)
        self.parts.append(f"### {title}{f' ({tag})' if tag else ''}\n\n{_markdown(df)}\n\n{caption}\n")

    def text(self, md: str):
        self.parts.append(md)

    def write(self, header: str) -> Path:
        path = self.out / "README.md"
        path.write_text(header + "\n\n" + "\n".join(self.parts))
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
    cases = cases[cases.family.isin(FAMILIES)]
    fields = fields[fields.family.isin(FAMILIES)]
    return cases, fields, systems


def _guess_family(system: str) -> str:
    if system == "rules":
        return "Rules"
    for prefix, fam in [("encoder", "Encoder"), ("nuextract", "NuExtract"), ("vllm", "Open LLM"), ("api", "API LLM"),
                        ("jev", "System One")]:
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


# ---------------------------------------------------------------- figures


def fig_headline(page: Page, cases: pd.DataFrame):
    plt = _plt()
    d = cases[cases.variant == "D"]
    order = _system_order(cases)
    sets = [s for s in SET_ORDER if s in set(d.test_set)]
    fig, axes = plt.subplots(1, len(sets), figsize=(2.15 * len(sets) + 1.6, 0.27 * len(order) + 1.4), sharey=True)
    y = {s: i for i, s in enumerate(reversed(order))}
    for ax, test_set in zip(axes, sets):
        g = d[d.test_set == test_set].groupby(["system", "family"]).score_correct.agg(["sum", "count"]).reset_index()
        for _, r in g.iterrows():
            lo, hi = _wilson(r["sum"], r["count"])
            acc = r["sum"] / r["count"]
            ax.plot([lo, hi], [y[r.system]] * 2, color=AXIS, lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter(acc, y[r.system], s=34, color=FAMILY_COLOR[r.family], edgecolor=SURFACE, linewidth=1.6, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_title(SET_LABEL[test_set], fontsize=8.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels([_short(s) for s in y])
    handles = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], markersize=6, label=f) for f in FAMILIES]
    fig.legend(handles=handles, loc="upper center", ncol=len(FAMILIES), bbox_to_anchor=(0.55, 1.02))
    fig.supxlabel("Final-score accuracy (OpenMedCalc on the extracted request), 95% Wilson CI", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    page.figure(fig, "fig02_headline", "Figure 2. Final-score accuracy by system and test set",
                "Primary endpoint 1 (plan §8.1). Rows are grouped by system family and sorted by accuracy on the "
                "primary sets (T2 ∪ T3). T1 and A409 are reported for comparability only (plan §5.3). "
                "Table 3 is the table view.")


def fig_generalisation(page: Page, cases: pd.DataFrame):
    plt = _plt()
    d = cases[cases.variant == "D"]
    order = _system_order(cases)
    t1 = d[d.test_set == "T1"].groupby("system").score_correct.mean()
    prim = d[d.test_set.isin(PRIMARY)].groupby("system").score_correct.mean()
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(order) + 1.2))
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
    page.figure(fig, "fig03_generalisation", "Figure 3. Generalisation gap: template text vs new wording",
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
    fig, ax = plt.subplots(figsize=(0.62 * len(calcs) + 2.8, 0.27 * len(order) + 1.3))
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
    page.figure(fig, "fig04_field_heatmap", "Figure 4. Field accuracy by system and calculator",
                "Primary endpoint 2 (plan §8.1), broken down by calculator. Scale starts at 50%. "
                "Table 5 gives the per-calculator numbers for the best system of each family.")


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
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(FAMILIES) + 0.6))
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
    page.figure(fig, "fig07_t4_robustness", "Figure 7. Robustness to controlled perturbations (T4)",
                "RQ4. Each cell is the paired change in final-score accuracy, T4 minus the same base case in T1, "
                "averaged over the systems in a family. Red is worse than on the original text.")


def fig_field_kind(page: Page, fields: pd.DataFrame):
    plt = _plt()
    import numpy as np

    f = fields[(fields.variant == "D") & fields.test_set.isin(PRIMARY)]
    kinds = ["bool", "enum", "int", "float"]
    acc = f.groupby(["family", "kind"]).correct.mean().unstack().reindex(index=FAMILIES, columns=kinds)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
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
    page.figure(fig, "fig05_field_kind", "Figure 5. Accuracy by field type",
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
    page.figure(fig, "fig06_assertion", "Figure 6. Documented vs undocumented criteria (T5)",
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
    page.figure(fig, "fig08_schema_variant", "Figure 8. Decision vs raw-facts schema (HAS-BLED)",
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
    page.figure(fig, "fig11_cost", "Figure 11. Accuracy against speed and deployment",
                "RQ5. Colour is where the system can run; this matters for patient data. The best system of each "
                "family is labelled. Table 9 adds GPU-hours.")


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
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
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
    page.figure(fig, "fig09_paper_arms", "Figure 9. Extract-then-compute vs the paper's end-to-end arms",
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
    fig, ax = plt.subplots(figsize=(7.6, 0.48 * len(FAMILIES) + 0.5))
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
    page.figure(fig, "fig10_error_taxonomy", "Figure 10. What kind of mistakes each family makes",
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
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
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
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(latest.confidence, bins) - 1, 0, 9)
    acc = latest.groupby(idx).correct.mean()
    n = latest.groupby(idx).size()
    ax2.bar(acc.index * 0.1 + 0.05, acc.values, width=0.08, color=FAMILY_COLOR["System One"], zorder=2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    _pct(ax2.yaxis)
    ax2.set_xlabel(f"Reported confidence ({_short(systems[0])})")
    ax2.set_ylabel("Field accuracy in bin")
    ax2.grid(axis="x", visible=False)
    for x_, v, k in zip(acc.index * 0.1 + 0.05, acc.values, n.values):
        if k < 30:
            ax2.text(x_, v + 0.02, f"n={k}", ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    page.figure(fig, "fig12_confidence", "Figure 12. Using reported confidence to decide what to automate",
                "Systems that return a confidence per answer (Jev) can defer uncertain fields to a human. Left: "
                "accuracy of the fields kept as more are deferred (risk-coverage). Right: accuracy by confidence "
                "bin; it should rise from left to right. Table 11 gives operating points.")


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
    page.table(pd.DataFrame(rows), "Table 11. Confidence-gated operating points",
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
    page.figure(fig, "fig13_a409_attribution", "Figure 13. Can a final-score error be traced to one field?",
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
    page.figure(fig, "fig14_cpu_baselines", "Figure 14. CPU baselines across test sets",
                "Real data. Perfect scores on template text (T1, A409) fall to the floor once the wording is new: "
                "the reason T2, T3 and T5 are primary.", real=True)


# ---------------------------------------------------------------- tables


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
    page.table(pd.DataFrame(rows), "Table 1. Test sets",
               "T5 is built locally from MedCalc-Bench Verified (CC-BY-SA) and is not committed. "
               "T2/T3 sizes are targets until clinician review is complete.", real=True)


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
    page.table(pd.DataFrame(rows), "Table 2. Systems under comparison",
               "From `cluster/killarney/models.tsv`. Encoders are fine-tuned on TRAIN; all LLM rows use "
               "JSON-schema constrained decoding unless marked otherwise.", real=True)


def table_primary(page: Page, cases: pd.DataFrame, fields: pd.DataFrame):
    d = cases[cases.variant == "D"]
    f = fields[fields.variant == "D"]
    rows = []
    for s in _system_order(cases):
        ds_ = d[d.system == s]
        prim = ds_[ds_.test_set.isin(PRIMARY)]
        lo, hi = _wilson(prim.score_correct.sum(), len(prim))
        fp = f[(f.system == s) & f.test_set.isin(PRIMARY)]
        macro = fp.groupby(["calculator", "field"]).correct.mean().groupby("calculator").mean().mean()
        row = {"system": _short(s), "family": ds_.family.iloc[0],
               "T1": ds_[ds_.test_set == "T1"].score_correct.mean(),
               "T2∪T3 score": f"{prim.score_correct.mean():.3f} [{lo:.3f}, {hi:.3f}]",
               "T2∪T3 macro field": macro,
               "T4": ds_[ds_.test_set == "T4"].score_correct.mean(),
               "T5": ds_[ds_.test_set == "T5"].score_correct.mean() if (ds_.test_set == "T5").any() else float("nan"),
               "valid": prim.valid.mean()}
        rows.append(row)
    page.table(pd.DataFrame(rows), "Table 3. Primary endpoints",
               "Score = final-score accuracy, OpenMedCalc on the extracted request. Macro field = per-field "
               "accuracy averaged within calculator, then across calculators. Valid = schema-valid outputs.")


def _best_per_family(cases: pd.DataFrame) -> Dict[str, str]:
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    acc = d.groupby(["family", "system"]).score_correct.mean().reset_index()
    return {f: g.sort_values("score_correct").system.iloc[-1] for f, g in acc.groupby("family")}


def table_pairwise(page: Page, cases: pd.DataFrame):
    best = _best_per_family(cases)
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    wide = d.pivot_table(index="id", columns="system", values="score_correct", aggfunc="first")
    tests, rows = {}, []
    fams = [f for f in FAMILIES if f in best]
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            sa, sb = best[a], best[b]
            pair = wide[[sa, sb]].dropna()
            res = stats.mcnemar_exact(pair[sa].astype(bool).tolist(), pair[sb].astype(bool).tolist())
            key = f"{a} vs {b}"
            tests[key] = res["p"]
            rows.append({"comparison": key, "systems": f"{_short(sa)} vs {_short(sb)}", "n": len(pair),
                         "only first right": res["a_only"], "only second right": res["b_only"],
                         "Δ accuracy": pair[sa].mean() - pair[sb].mean(), "p (exact McNemar)": res["p"]})
    adj = stats.holm(tests)
    for r in rows:
        r["p (Holm)"] = adj[r["comparison"]]
        for k in ("p (exact McNemar)", "p (Holm)"):
            r[k] = "<0.001" if r[k] < 0.001 else f"{r[k]:.3f}"
    page.table(pd.DataFrame(rows), "Table 4. Pairwise comparisons of the best system per family",
               "Family (i) of the pre-specified comparisons (plan §9), on T2 ∪ T3, paired by case.")


def table_per_calculator(page: Page, cases: pd.DataFrame):
    best = _best_per_family(cases)
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY) & cases.system.isin(best.values())]
    acc = d.groupby(["calculator", "system"]).score_correct.mean().unstack()
    acc = acc[[best[f] for f in FAMILIES if f in best]]
    acc.columns = [f"{f}: {_short(best[f])}" for f in FAMILIES if f in best]
    acc.insert(0, "n", d.groupby("calculator").id.nunique())
    page.table(acc.reset_index(), "Table 5. Final-score accuracy by calculator (best system per family, T2 ∪ T3)",
               "Per-calculator n is about 200 per system, so these rows are descriptive (plan §9).")


def table_band(page: Page, cases: pd.DataFrame):
    d = cases[(cases.variant == "D") & cases.test_set.isin(PRIMARY)]
    g = d.groupby("system").agg(score=("score_correct", "mean"), band=("band_correct", "mean"),
                                all_fields=("all_fields_correct", "mean"), mae=("abs_error", "mean"))
    g = g.reindex(_system_order(cases)).reset_index()
    g["system"] = g.system.map(_short)
    g.columns = ["system", "score accuracy", "risk-band agreement", "all fields correct", "mean |score error|"]
    page.table(g, "Table 6. Clinical agreement: risk bands, all-fields rate, score error",
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
    page.table(g, "Table 9. Operational metrics",
               "Plan §8.4. Real latency comes from `latency_s` in each prediction; GPU-hours scale it by the GPUs "
               "in `models.tsv`; API cost is added from the provider's invoice.")


def table_mixed_effects_shell(page: Page):
    terms = ["(Intercept)", "system [each vs rules]", "schema variant R", "field type: enum", "field type: int",
             "field type: float", "test set: T2", "test set: T3", "test set: T4", "test set: T5",
             "system × test set (one row per pair)", "random intercept SD: calculator", "random intercept SD: case",
             "random intercept SD: field"]
    df = pd.DataFrame({"term": terms, "log-odds [95% CI]": "—", "odds ratio": "—", "p": "—"})
    page.table(df, "Table 10. Mixed-effects logistic model of field correctness",
               "Plan §9: correct ~ system + schema_variant + field_type + test_set + system:test_set + "
               "(1|calculator) + (1|case) + (1|field). Fitted on real field-level rows only; no mock values.",
               shell=True)


def _try(page: Page, fn, *args):
    """Render one figure or table; record why if the data for it does not exist yet."""
    try:
        return fn(page, *args)
    except Exception as e:  # noqa: BLE001 - a missing set must not stop the rest of the page
        name = fn.__name__.replace("_", " ")
        page.text(f"_{name}: not rendered ({type(e).__name__}: {str(e).splitlines()[0][:120]})._\n")
        return None


def write_all(eval_dir: Path, out: Path, mock: bool) -> Path:
    cases, fields, systems = load(eval_dir)
    page = Page(out, mock)
    page.text("## Design\n\n```mermaid\nflowchart LR\n  N[Clinical text] --> X[Extractor<br/>rules · encoder · NuExtract · LLM]\n"
              "  X --> R[Typed request<br/>Pydantic, validated]\n  R --> C[OpenMedCalc]\n  C --> S[Score]\n"
              "  R -. per-field grading .-> G1[(Gold request)]\n  S -. score grading .-> G2[(Gold score)]\n```\n\n"
              "Figure 1. Every system fills the same typed request; the calculator is deterministic, so all "
              "errors are extraction errors.\n")
    page.text("## Data\n")
    _try(page, table_datasets)
    _try(page, table_systems, systems)
    page.text("## Primary endpoints\n")
    _try(page, fig_headline, cases)
    _try(page, table_primary, cases, fields)
    _try(page, fig_generalisation, cases)
    _try(page, table_pairwise, cases)
    page.text("## Where systems succeed and fail\n")
    _try(page, fig_heatmap, fields, cases)
    _try(page, table_per_calculator, cases)
    _try(page, fig_field_kind, fields)
    _try(page, fig_assertion, fields)
    _try(page, table_band, cases)
    page.text("## Robustness, schema design and the original study\n")
    _try(page, fig_t4, cases)
    _try(page, fig_variant, cases)
    arms = _try(page, fig_paper_arms, eval_dir)
    if arms is not None:
        t = arms.pivot_table(index="arm", columns="model", values="acc", sort=False).reset_index()
        page.table(t, "Table 7. Paper arms vs extract-then-compute (T1)", "Table view of Figure 9.")
    page.text("## Errors, cost and the statistical model\n")
    share = _try(page, fig_errors, eval_dir)
    if share is not None:
        page.table(share.reset_index().rename(columns={"index": "family"}),
                   "Table 8. Error codes by family (share of annotated field errors)", "Table view of Figure 10.")
    _try(page, fig_cost, cases, systems)
    _try(page, table_ops, cases, systems)
    _try(page, table_mixed_effects_shell)
    page.text("## Confidence and selective automation\n")
    _try(page, fig_selective, fields)
    _try(page, table_confidence, fields, cases)
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
