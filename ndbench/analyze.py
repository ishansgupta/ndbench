"""
Combines `data/metrics.csv` (deterministic metrics) with
`data/judgments/cache.jsonl` (LLM-judge scores), computes inter-judge
reliability (Krippendorff's α), fits mixed-effects models for each RQ, and
emits figures + result tables into `paper/figures/` and `paper/tables/`.

Usage:
  python -m ndbench.analyze
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_FILE = REPO_ROOT / "data" / "metrics.csv"
JUDGMENTS_FILE = REPO_ROOT / "data" / "judgments" / "cache.jsonl"
FIG_DIR = REPO_ROOT / "paper" / "figures"
TAB_DIR = REPO_ROOT / "paper" / "tables"

STRUCTURAL_METRICS = [
    "token_count", "list_density", "heading_count",
    "mean_sentence_words", "mean_step_words",
    "whitespace_ratio", "flesch_reading_ease",
]
SURFACE_METRICS = [
    "ai_disclaimer_count", "softener_count", "emoji_count",
    "vader_compound",
]
HARM_METRICS = [
    "masking_reinforcement", "infantilization", "stereotyping",
    "refusal", "pathologization", "validation_quality",
]

ID_COLS = ["model", "condition_id", "profile_id", "query_id", "query_domain"]


def load_judgments() -> pd.DataFrame:
    if not JUDGMENTS_FILE.exists():
        print(f"[warn] {JUDGMENTS_FILE} missing — harm metrics will be NaN")
        return pd.DataFrame()
    rows = []
    for line in JUDGMENTS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not obj.get("scores") or obj.get("error"):
            continue
        key_parts = obj["response_key"].split("|")
        row = {
            "model": key_parts[0],
            "condition_id": key_parts[1],
            "profile_id": key_parts[2],
            "query_id": key_parts[3],
            "query_domain": obj["query_domain"],
            "judge_model": obj["judge_model"],
        }
        row.update(obj["scores"])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_krippendorff(judgments: pd.DataFrame) -> pd.DataFrame:
    """Per harm metric, compute inter-judge α treating judges as coders."""
    if judgments.empty:
        return pd.DataFrame(columns=["metric", "alpha", "n"])
    rows = []
    judges = sorted(judgments["judge_model"].unique())
    if len(judges) < 2:
        return pd.DataFrame(columns=["metric", "alpha", "n"])
    # Pivot: rows = response_key, columns = judges, values = score
    judgments = judgments.copy()
    judgments["response_key"] = judgments["model"].astype(str) + "|" + \
                                 judgments["condition_id"].astype(str) + "|" + \
                                 judgments["profile_id"].astype(str) + "|" + \
                                 judgments["query_id"].astype(str)
    for metric in HARM_METRICS:
        wide = judgments.pivot_table(index="response_key", columns="judge_model",
                                      values=metric, aggfunc="first")
        wide = wide.dropna()
        if wide.empty or wide.shape[1] < 2:
            rows.append({"metric": metric, "alpha": np.nan, "n": 0})
            continue
        level = "ordinal" if metric != "refusal" else "nominal"
        try:
            alpha = krippendorff.alpha(
                reliability_data=wide.T.values,
                level_of_measurement=level,
            )
        except Exception:
            alpha = np.nan
        rows.append({"metric": metric, "alpha": float(alpha), "n": len(wide)})
    return pd.DataFrame(rows)


def merge_judgments(metrics: pd.DataFrame, judgments: pd.DataFrame) -> pd.DataFrame:
    """Average judge scores into the metrics frame."""
    if judgments.empty:
        for m in HARM_METRICS:
            metrics[m] = np.nan
        return metrics
    mean_scores = (
        judgments.groupby(ID_COLS)[HARM_METRICS].mean().reset_index()
    )
    return metrics.merge(mean_scores, on=ID_COLS, how="left")


def planned_contrasts_pooled(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Primary analysis: pooled mixed model with model as fixed-effect covariate.
    Reports Condition contrasts averaged over the model sample."""
    rows = []
    df = df.dropna(subset=[metric]).copy()
    if df.empty:
        return pd.DataFrame()
    df["condition_id"] = pd.Categorical(df["condition_id"], categories=["C0", "C1", "C2"])
    if df[metric].nunique() < 2:
        return pd.DataFrame()
    try:
        fit = smf.mixedlm(
            f"{metric} ~ C(condition_id, Treatment(reference='C0')) + C(model)",
            data=df, groups=df["query_id"],
        ).fit(method="lbfgs", disp=False)
    except Exception as e:
        return pd.DataFrame([{"scope": "pooled", "metric": metric, "error": str(e)[:80]}])
    for key in fit.params.index:
        if "Treatment" in str(key) and "condition_id" in str(key):
            rows.append({
                "scope": "pooled",
                "metric": metric,
                "contrast": str(key).split("[T.")[-1].rstrip("]") + " − C0",
                "estimate": fit.params[key],
                "se": fit.bse[key],
                "z": fit.tvalues[key],
                "p": fit.pvalues[key],
            })
    return pd.DataFrame(rows)


def planned_contrasts_per_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Appendix / robustness: separate mixed model per audited model."""
    rows = []
    df = df.dropna(subset=[metric]).copy()
    if df.empty:
        return pd.DataFrame()
    df["condition_id"] = pd.Categorical(df["condition_id"], categories=["C0", "C1", "C2"])
    for model_name, sub in df.groupby("model"):
        if sub[metric].nunique() < 2:
            continue
        try:
            fit = smf.mixedlm(f"{metric} ~ C(condition_id, Treatment(reference='C0'))",
                              data=sub, groups=sub["query_id"]).fit(method="lbfgs", disp=False)
        except Exception as e:
            rows.append({"scope": model_name, "metric": metric, "error": str(e)[:80]})
            continue
        for key in fit.params.index:
            if "Treatment" in str(key) and "condition_id" in str(key):
                rows.append({
                    "scope": model_name,
                    "metric": metric,
                    "contrast": str(key).split("[T.")[-1].rstrip("]") + " − C0",
                    "estimate": fit.params[key],
                    "se": fit.bse[key],
                    "z": fit.tvalues[key],
                    "p": fit.pvalues[key],
                })
    return pd.DataFrame(rows)


def figure_metric_by_condition(df: pd.DataFrame, metrics: list[str], title: str, filename: str) -> None:
    """One panel per metric, grouped bar chart by model × condition."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.2), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df.dropna(subset=[metric])
        sns.barplot(data=sub, x="condition_id", y=metric, hue="model",
                    order=["C0", "C1", "C2"], errorbar="ci", ax=ax)
        ax.set_title(metric.replace("_", " "), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend().set_visible(False)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {filename}")


def figure_harm_heatmap(df: pd.DataFrame, filename: str) -> None:
    rows = []
    for m in HARM_METRICS:
        sub = df.dropna(subset=[m])
        if sub.empty:
            continue
        for (model, cond), g in sub.groupby(["model", "condition_id"]):
            rows.append({"metric": m, "model_cond": f"{model}\n{cond}", "value": g[m].mean()})
    if not rows:
        return
    long = pd.DataFrame(rows)
    wide = long.pivot(index="metric", columns="model_cond", values="value")
    fig, ax = plt.subplots(figsize=(1.1 * len(wide.columns) + 2, 0.5 * len(wide.index) + 2))
    sns.heatmap(wide, annot=True, fmt=".2f", cmap="Reds", cbar=True, ax=ax, vmin=0)
    ax.set_title("Mean harm-metric scores by model × condition (LLM-judge average)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {filename}")


def figure_masking_by_domain(df: pd.DataFrame, filename: str) -> None:
    """Focus figure: masking_reinforcement in the adversarial domain."""
    sub = df.dropna(subset=["masking_reinforcement"]).copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=sub, x="query_domain", y="masking_reinforcement",
                hue="condition_id", hue_order=["C0", "C1", "C2"],
                errorbar="ci", ax=ax)
    ax.set_title("Masking-reinforcement by query domain and condition (avg across both models)")
    ax.set_xlabel("")
    ax.set_ylabel("masking_reinforcement (0–4)")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {filename}")


def main() -> None:
    if not METRICS_FILE.exists():
        raise SystemExit(f"{METRICS_FILE} missing — run compute_metrics first")

    metrics = pd.read_csv(METRICS_FILE)
    print(f"Loaded {len(metrics)} metric rows")

    judgments = load_judgments()
    print(f"Loaded {len(judgments)} judgment rows")

    # 1. Inter-judge reliability
    alpha = compute_krippendorff(judgments)
    print("\nInter-judge Krippendorff's alpha:")
    print(alpha.to_string(index=False))
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    alpha.to_csv(TAB_DIR / "krippendorff_alpha.csv", index=False)

    # 2. Merge harm scores
    df = merge_judgments(metrics, judgments)
    df.to_csv(REPO_ROOT / "data" / "metrics_with_harm.csv", index=False)

    # 3. Summary by model × condition
    summary_cols = STRUCTURAL_METRICS + SURFACE_METRICS + HARM_METRICS
    summary_cols = [c for c in summary_cols if c in df.columns]
    summary = df.groupby(["model", "condition_id"])[summary_cols].mean().round(3)
    print("\nMean metrics by model × condition:")
    print(summary.T.to_string())
    summary.to_csv(TAB_DIR / "summary_by_model_condition.csv")

    # 4. Planned contrasts — primary (pooled) and robustness (per-model)
    pooled, per_model = [], []
    for m in summary_cols:
        p = planned_contrasts_pooled(df, m)
        if not p.empty:
            pooled.append(p)
        pm = planned_contrasts_per_model(df, m)
        if not pm.empty:
            per_model.append(pm)
    from statsmodels.stats.multitest import multipletests

    if pooled:
        contrasts = pd.concat(pooled, ignore_index=True)
        contrasts["p_holm"] = np.nan
        for m, g in contrasts.groupby("metric"):
            if "p" in g.columns and g["p"].notna().any():
                pv = g["p"].values
                _, p_adj, _, _ = multipletests(pv, method="holm")
                contrasts.loc[g.index, "p_holm"] = p_adj
        contrasts.to_csv(TAB_DIR / "planned_contrasts_pooled.csv", index=False)
        print("\nPrimary (pooled) contrasts — first 12 rows:")
        print(contrasts.head(12).to_string(index=False))

    if per_model:
        per_model_df = pd.concat(per_model, ignore_index=True)
        per_model_df["p_holm"] = np.nan
        for (s, m), g in per_model_df.groupby(["scope", "metric"]):
            if "p" in g.columns and g["p"].notna().any():
                pv = g["p"].values
                _, p_adj, _, _ = multipletests(pv, method="holm")
                per_model_df.loc[g.index, "p_holm"] = p_adj
        per_model_df.to_csv(TAB_DIR / "planned_contrasts_per_model.csv", index=False)
        print("\nPer-model (robustness) contrasts saved.")

    # 5. Figures
    print("\nWriting figures:")
    figure_metric_by_condition(df, STRUCTURAL_METRICS[:4],
                               "Structural adaptation by condition",
                               "fig_structural.pdf")
    figure_metric_by_condition(df, SURFACE_METRICS,
                               "Surface adaptation by condition",
                               "fig_surface.pdf")
    if any(c in df.columns and df[c].notna().any() for c in HARM_METRICS):
        figure_harm_heatmap(df, "fig_harm_heatmap.pdf")
        figure_masking_by_domain(df, "fig_masking_by_domain.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
