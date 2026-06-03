"""
src/analysis.py
Implements design.md § Analysis Methods and requirements.md § Analysis.

Reads data/processed/results.csv; writes:
  results/aggregated_results.csv
  results/statistical_results.csv
  results/plots/*.png

All analyses filter per-field on non-999 values (design.md § Missing data).
Q1=999 rows are dropped globally before any analysis.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print(
        "WARN: pingouin not installed — Welch ANOVA uses scipy fallback; "
        "Games-Howell skipped. Install with: pip install pingouin",
        file=sys.stderr,
    )

_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
_RESULTS_DIR   = Path(__file__).parent.parent / "results"

CONTINUOUS      = ["severity_score", "impairment_score", "treatment_score", "minimizing_score"]
CATEGORICAL     = ["diagnosis_yes_no", "treatment_type_profile", "framing_label"]
VARIANT_COLS    = ["prompt_type", "gender", "age_group", "severity_level", "model"]
DEMO_CONDITIONS = ["base", "mitigation"]          # conditions with full demographic header
N_PRIMARY_TESTS = 12                              # Bonferroni k: 4 outcomes × 3 grouping dims

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    """
    Load results.csv and drop rows where diagnosis_yes_no == 999.
    (design.md § Missing data: Q1=999 → row is invalid and excluded from analysis.)
    """
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["diagnosis_yes_no"] != 999].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with diagnosis_yes_no=999")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Descriptive statistics  →  aggregated_results.csv
# ──────────────────────────────────────────────────────────────────────────────

def _ci95(s: pd.Series) -> float:
    n = s.count()
    return float(1.96 * s.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def _descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per variant (VARIANT_COLS), compute:
      - mean, std, min, max, n for each continuous outcome (non-999 values only)
      - depression_rate and q1_agreement_rate for diagnosis_yes_no
      - frequency counts / pct for treatment_type_profile and framing_label
    Returns a wide-format summary DataFrame.
    """
    base = df.groupby(VARIANT_COLS, as_index=False).size().rename(columns={"size": "n_runs"})

    # ── Continuous outcomes ──────────────────────────────────────────────────
    for col in CONTINUOUS:
        valid = df[df[col] != 999].groupby(VARIANT_COLS)[col]
        stats = valid.agg(["mean", "std", "min", "max", "count"]).add_prefix(f"{col}_")
        stats = stats.rename(columns={f"{col}_count": f"{col}_n"}).reset_index()
        base = base.merge(stats, on=VARIANT_COLS, how="left")

    # ── Q1: depression rate and agreement rate ───────────────────────────────
    def _agreement(series: pd.Series) -> float:
        vals = series[series != 999]
        if vals.empty:
            return float("nan")
        return float(vals.value_counts().max() / len(vals))

    q1 = df.groupby(VARIANT_COLS)["diagnosis_yes_no"].agg(
        q1_depression_rate=lambda s: s[s != 999].mean(),
        q1_n=lambda s: (s != 999).sum(),
        q1_agreement_rate=_agreement,
    ).reset_index()
    base = base.merge(q1, on=VARIANT_COLS, how="left")

    # ── Q4b: treatment type profile frequencies ──────────────────────────────
    q4b_valid = df[df["treatment_type_profile"] != 999]
    if not q4b_valid.empty:
        q4b = (
            q4b_valid.groupby(VARIANT_COLS + ["treatment_type_profile"])
            .size()
            .rename("q4b_count")
            .reset_index()
        )
        totals = q4b_valid.groupby(VARIANT_COLS).size().rename("q4b_total").reset_index()
        q4b   = q4b.merge(totals, on=VARIANT_COLS)
        q4b["q4b_pct"] = 100 * q4b["q4b_count"] / q4b["q4b_total"]
        # Pivot so one row per variant
        q4b_piv = q4b.pivot_table(
            index=VARIANT_COLS, columns="treatment_type_profile",
            values="q4b_pct", fill_value=0,
        ).add_prefix("q4b_pct_").reset_index()
        base = base.merge(q4b_piv, on=VARIANT_COLS, how="left")

    # ── Q6: framing label frequencies ───────────────────────────────────────
    framing_valid = df[df["framing_label"] != 999]
    if not framing_valid.empty:
        frm = (
            framing_valid.groupby(VARIANT_COLS + ["framing_label"])
            .size()
            .rename("frm_count")
            .reset_index()
        )
        frm_totals = framing_valid.groupby(VARIANT_COLS).size().rename("frm_total").reset_index()
        frm = frm.merge(frm_totals, on=VARIANT_COLS)
        frm["frm_pct"] = 100 * frm["frm_count"] / frm["frm_total"]
        frm_piv = frm.pivot_table(
            index=VARIANT_COLS, columns="framing_label",
            values="frm_pct", fill_value=0,
        ).add_prefix("frm_pct_").reset_index()
        base = base.merge(frm_piv, on=VARIANT_COLS, how="left")

    return base


# ──────────────────────────────────────────────────────────────────────────────
# Statistical tests  →  statistical_results.csv
# ──────────────────────────────────────────────────────────────────────────────

def _welch_anova_scipy(groups: list[np.ndarray]) -> tuple[float, float, float, float]:
    """
    Welch's one-way ANOVA, implemented from the Welch (1951) formula.
    Returns (F, df_numerator, df_denominator, p_value).
    """
    k   = len(groups)
    n   = np.array([len(g) for g in groups], dtype=float)
    mu  = np.array([np.mean(g) for g in groups])
    var = np.array([np.var(g, ddof=1) for g in groups])

    if np.any(var == 0) or np.any(n < 2):
        return float("nan"), float(k - 1), float("nan"), float("nan")

    w         = n / var
    W         = w.sum()
    grand_mu  = (w * mu).sum() / W
    numerator = (w * (mu - grand_mu) ** 2).sum() / (k - 1)
    S         = ((1 - w / W) ** 2 / (n - 1)).sum()
    denom     = 1 + (2 * (k - 2) / (k ** 2 - 1)) * S
    F         = numerator / denom
    df_num    = float(k - 1)
    df_den    = (k ** 2 - 1) / (3 * S)
    p         = float(1 - ss.f.cdf(F, df_num, df_den))
    return float(F), df_num, float(df_den), p


def _run_welch_anova(data: pd.DataFrame, dv: str, between: str) -> dict:
    """Run one Welch's ANOVA and return a result dict."""
    groups_map = {g: grp[dv].dropna().values for g, grp in data.groupby(between)}
    groups     = [v for v in groups_map.values() if len(v) >= 2]
    if len(groups) < 2:
        return {
            "test_type": "welch_anova", "outcome": dv, "grouping_var": between,
            "group1": None, "group2": None, "statistic": float("nan"),
            "df_num": float("nan"), "df_den": float("nan"),
            "p_value": float("nan"), "p_corrected": float("nan"),
            "significant_corrected": False, "note": "insufficient data",
        }

    if HAS_PINGOUIN:
        try:
            aov = pg.welch_anova(data=data, dv=dv, between=between)
            # Column is "p_unc" (underscore) in pingouin >= 0.5.4
            p_col = "p_unc" if "p_unc" in aov.columns else "p-unc"
            F   = float(aov["F"].iloc[0])
            df1 = float(aov["ddof1"].iloc[0])
            df2 = float(aov["ddof2"].iloc[0])
            p   = float(aov[p_col].iloc[0])
        except Exception as exc:
            return {"test_type": "welch_anova", "outcome": dv, "grouping_var": between,
                    "note": f"pingouin error: {exc}", "p_value": float("nan"),
                    "statistic": float("nan"), "df_num": float("nan"), "df_den": float("nan"),
                    "group1": None, "group2": None,
                    "p_corrected": float("nan"), "significant_corrected": False}
    else:
        F, df1, df2, p = _welch_anova_scipy(groups)

    return {
        "test_type": "welch_anova", "outcome": dv, "grouping_var": between,
        "group1": None, "group2": None,
        "statistic": round(F, 4), "df_num": round(df1, 2), "df_den": round(df2, 2),
        "p_value": round(p, 4), "p_corrected": None, "significant_corrected": None,
        "note": "pingouin" if HAS_PINGOUIN else "scipy-fallback",
    }


def _run_games_howell(data: pd.DataFrame, dv: str, between: str) -> list[dict]:
    """Games-Howell post-hoc — requires pingouin."""
    if not HAS_PINGOUIN:
        return []
    try:
        ph = pg.pairwise_gameshowell(data=data, dv=dv, between=between)
        rows = []
        for _, r in ph.iterrows():
            rows.append({
                "test_type": "games_howell", "outcome": dv, "grouping_var": between,
                "group1": r["A"], "group2": r["B"],
                "statistic": round(float(r["T"]), 4),
                "df_num": None, "df_den": round(float(r["df"]), 2),
                "p_value": round(float(r["pval"]), 4),
                "p_corrected": round(float(r["pval"]), 4),  # GH already adjusts
                "significant_corrected": float(r["pval"]) < 0.05,
                "note": "games-howell",
            })
        return rows
    except Exception as exc:
        return [{"test_type": "games_howell", "outcome": dv, "grouping_var": between,
                 "note": f"error: {exc}", "p_value": float("nan"),
                 "statistic": float("nan"), "group1": None, "group2": None,
                 "df_num": None, "df_den": None,
                 "p_corrected": float("nan"), "significant_corrected": False}]


def _run_chi_fisher(data: pd.DataFrame, outcome: str, grouping_var: str) -> dict:
    """Chi-square or Fisher's exact (2×2 only) across grouping_var."""
    valid = data[data[outcome] != 999]
    if valid.empty or valid[grouping_var].nunique() < 2:
        return {"test_type": "chi_square", "outcome": outcome,
                "grouping_var": grouping_var, "group1": None, "group2": None,
                "statistic": float("nan"), "df_num": float("nan"), "df_den": None,
                "p_value": float("nan"), "p_corrected": None, "significant_corrected": None,
                "note": "insufficient data"}

    ct = pd.crosstab(valid[grouping_var], valid[outcome])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p, dof, expected = ss.chi2_contingency(ct)

    if (expected < 5).any() and ct.shape == (2, 2):
        odds, p_fisher = ss.fisher_exact(ct.values)
        return {
            "test_type": "fisher_exact", "outcome": outcome,
            "grouping_var": grouping_var, "group1": None, "group2": None,
            "statistic": round(float(odds), 4), "df_num": None, "df_den": None,
            "p_value": round(float(p_fisher), 4), "p_corrected": None,
            "significant_corrected": p_fisher < 0.05,
            "note": "fisher_exact (expected cell < 5)",
        }

    note = "chi_square (some expected < 5; table > 2x2)" if (expected < 5).any() else "chi_square"
    return {
        "test_type": "chi_square", "outcome": outcome,
        "grouping_var": grouping_var, "group1": None, "group2": None,
        "statistic": round(float(chi2), 4), "df_num": int(dof), "df_den": None,
        "p_value": round(float(p), 4), "p_corrected": None,
        "significant_corrected": p < 0.05, "note": note,
    }


def _run_welch_ttest(a: pd.Series, b: pd.Series, dv: str, label_a: str, label_b: str,
                     note: str) -> dict:
    """Welch's two-sample t-test (equal_var=False)."""
    a_clean = a[a != 999].dropna()
    b_clean = b[b != 999].dropna()
    if len(a_clean) < 2 or len(b_clean) < 2:
        return {"test_type": "welch_ttest", "outcome": dv, "grouping_var": "prompt_type",
                "group1": label_a, "group2": label_b,
                "statistic": float("nan"), "df_num": None, "df_den": float("nan"),
                "p_value": float("nan"), "p_corrected": None, "significant_corrected": None,
                "note": f"insufficient data — {note}"}
    t, p = ss.ttest_ind(a_clean, b_clean, equal_var=False)
    return {
        "test_type": "welch_ttest", "outcome": dv, "grouping_var": "neutral_vs_base",
        "group1": label_a, "group2": label_b,
        "statistic": round(float(t), 4), "df_num": None, "df_den": None,
        "p_value": round(float(p), 4), "p_corrected": None,
        "significant_corrected": p < 0.05, "note": note,
    }


def _inferential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all inferential tests and return a long-format results DataFrame.
    Bonferroni correction applied across the 12 primary Welch ANOVA tests
    (4 continuous outcomes × 3 grouping dimensions).
    """
    records: list[dict] = []

    # ── 1. Welch ANOVA: 4 outcomes × 3 grouping dims (base + mitigation) ────
    demo = df[df["prompt_type"].isin(DEMO_CONDITIONS)].copy()
    anova_rows: list[dict] = []
    for outcome in CONTINUOUS:
        valid = demo[demo[outcome] != 999]
        for grp_var in ["gender", "age_group", "prompt_type"]:
            anova_rows.append(_run_welch_anova(valid, outcome, grp_var))

    # Bonferroni correction across N_PRIMARY_TESTS = 12
    for row in anova_rows:
        raw_p = row["p_value"]
        if not np.isnan(raw_p):
            p_corr = min(raw_p * N_PRIMARY_TESTS, 1.0)
            row["p_corrected"]          = round(p_corr, 4)
            row["significant_corrected"] = p_corr < 0.05

    records.extend(anova_rows)

    # ── 2. Games-Howell post-hoc for significant ANOVA results ───────────────
    for row in anova_rows:
        if row.get("significant_corrected"):
            outcome  = row["outcome"]
            grp_var  = row["grouping_var"]
            valid    = demo[demo[outcome] != 999]
            records.extend(_run_games_howell(valid, outcome, grp_var))

    # ── 3. Chi-square / Fisher for categorical outcomes ───────────────────────
    for outcome in CATEGORICAL:
        for grp_var in ["gender", "age_group"]:
            records.append(_run_chi_fisher(demo, outcome, grp_var))

    # ── 4. Neutral vs. base Welch t-tests (per severity level per outcome) ───
    neutral_conds = {
        "neutral_full":   ("neutral_full",   None,     None),
        "neutral_age":    ("neutral_age",    None,     None),
        "neutral_gender": ("neutral_gender", None,     None),
    }
    for sev in df["severity_level"].unique():
        base_sev = df[(df["prompt_type"] == "base") & (df["severity_level"] == sev)]
        for nc_name, (nc_type, _, _) in neutral_conds.items():
            nc_sev = df[(df["prompt_type"] == nc_type) & (df["severity_level"] == sev)]
            for outcome in CONTINUOUS:
                records.append(_run_welch_ttest(
                    base_sev[outcome], nc_sev[outcome],
                    outcome, "base", nc_name,
                    f"severity={sev} | {nc_name} vs base",
                ))

    # ── 5. Intersectionality: two-way ANOVA (gender × age on base condition) ─
    base_only = df[df["prompt_type"] == "base"].copy()
    for outcome in CONTINUOUS:
        valid = base_only[base_only[outcome] != 999]
        if HAS_PINGOUIN and len(valid) >= 10:
            try:
                aov2 = pg.anova(
                    data=valid, dv=outcome,
                    between=["gender", "age_group"], ss_type=3,
                )
                for _, r in aov2.iterrows():
                    source = str(r["Source"])
                    if source == "Residual":
                        continue
                    records.append({
                        "test_type": "two_way_anova", "outcome": outcome,
                        "grouping_var": source,
                        "group1": "gender", "group2": "age_group",
                        "statistic": round(float(r.get("F", float("nan"))), 4),
                        "df_num": int(r.get("DF", 0)),
                        "df_den": None,
                        "p_value": round(float(r.get("p-unc", float("nan"))), 4),
                        "p_corrected": None,
                        "significant_corrected": float(r.get("p-unc", 1)) < 0.05,
                        "note": "two-way ANOVA (not strictly Welch; standard Type-III SS)",
                    })
            except Exception as exc:
                records.append({
                    "test_type": "two_way_anova", "outcome": outcome,
                    "grouping_var": "gender*age_group",
                    "group1": "gender", "group2": "age_group",
                    "statistic": float("nan"), "df_num": None, "df_den": None,
                    "p_value": float("nan"), "p_corrected": None,
                    "significant_corrected": False,
                    "note": f"error: {exc}",
                })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Visualizations  →  results/plots/*.png
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _plot_scores_by_group(df: pd.DataFrame, group_col: str,
                           plots_dir: Path) -> None:
    """Grouped bar chart: mean Q2/Q3/Q4a/Q5 per group, with 95% CI error bars."""
    demo = df[df["prompt_type"].isin(DEMO_CONDITIONS)]
    labels = {
        "severity_score":   "Q2 Severity (0–10)",
        "impairment_score": "Q3 Impairment (0–10)",
        "treatment_score":  "Q4a Treatment (0–10)",
        "minimizing_score": "Q5 Minimizing (0–10)",
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, col in zip(axes, CONTINUOUS):
        valid = demo[demo[col] != 999]
        if valid.empty:
            ax.set_title(labels[col])
            continue
        grp  = valid.groupby(group_col)[col]
        means = grp.mean()
        errs  = grp.apply(_ci95)
        x     = np.arange(len(means))
        ax.bar(x, means, yerr=errs, capsize=4, color=sns.color_palette("muted", len(means)))
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(labels[col])
        ax.set_ylim(0, 10)
        ax.set_ylabel("Mean score")

    fig.suptitle(f"Mean scores by {group_col}", fontsize=11, y=1.01)
    _save(fig, plots_dir / f"scores_by_{group_col}.png")


def _plot_base_vs_mitigation(df: pd.DataFrame, plots_dir: Path) -> None:
    """Side-by-side bar chart: base vs. mitigation per gender per outcome."""
    data = df[df["prompt_type"].isin(DEMO_CONDITIONS)].copy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, col in zip(axes, CONTINUOUS):
        valid = data[data[col] != 999]
        if valid.empty:
            continue
        grp   = valid.groupby(["gender", "prompt_type"])[col].mean().unstack("prompt_type")
        grp.plot(kind="bar", ax=ax, capsize=3, legend=(col == CONTINUOUS[0]))
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylim(0, 10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Base vs. Mitigation by Gender", fontsize=11, y=1.01)
    _save(fig, plots_dir / "base_vs_mitigation.png")


def _plot_neutral_vs_base(df: pd.DataFrame, plots_dir: Path) -> None:
    """Bar chart: neutral conditions vs. base per severity level per outcome."""
    conditions = ["base", "neutral_full", "neutral_age", "neutral_gender"]
    data = df[df["prompt_type"].isin(conditions)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, col in zip(axes, CONTINUOUS):
        valid = data[data[col] != 999]
        if valid.empty:
            continue
        grp = valid.groupby(["severity_level", "prompt_type"])[col].mean().unstack("prompt_type")
        grp.plot(kind="bar", ax=ax, legend=(col == CONTINUOUS[0]))
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("severity")
        ax.set_ylim(0, 10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Neutral Conditions vs. Base by Severity", fontsize=11, y=1.01)
    _save(fig, plots_dir / "neutral_vs_base.png")


def _plot_q1_diagnosis_rate(df: pd.DataFrame, plots_dir: Path) -> None:
    """Stacked bar: diagnosis yes/no proportion per gender × prompt_type."""
    demo  = df[df["prompt_type"].isin(DEMO_CONDITIONS) & (df["diagnosis_yes_no"] != 999)]
    if demo.empty:
        return
    rate  = demo.groupby(["gender", "prompt_type"])["diagnosis_yes_no"].mean().rename("depression_rate")
    rate  = rate.reset_index()
    rate["no_depression"] = 1 - rate["depression_rate"]

    fig, ax = plt.subplots(figsize=(10, 4))
    piv = rate.pivot_table(index=["gender", "prompt_type"], values=["depression_rate", "no_depression"])
    piv[["depression_rate", "no_depression"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#e06c75", "#98c379"], legend=True,
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Q1 Depression Diagnosis Rate by Gender & Prompt Type")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(["Depression detected", "Not detected"], loc="upper right")
    _save(fig, plots_dir / "q1_diagnosis_rate.png")


def _plot_q4b_treatment_type(df: pd.DataFrame, plots_dir: Path) -> None:
    """Heatmap: treatment type profile distribution (gender × category)."""
    demo  = df[df["prompt_type"].isin(DEMO_CONDITIONS) & (df["treatment_type_profile"] != 999)]
    if demo.empty:
        return
    ct = pd.crosstab(demo["gender"], demo["treatment_type_profile"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={"label": "%"})
    ax.set_title("Q4b Treatment Type Profile by Gender (%)")
    ax.set_ylabel("Gender")
    ax.set_xlabel("Treatment Category")
    _save(fig, plots_dir / "q4b_treatment_type.png")


def _plot_q6_framing(df: pd.DataFrame, plots_dir: Path) -> None:
    """Stacked bar: framing label distribution per gender."""
    demo  = df[df["prompt_type"].isin(DEMO_CONDITIONS) & (df["framing_label"] != 999)]
    if demo.empty:
        return
    ct = pd.crosstab(demo["gender"], demo["framing_label"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Q6 Person Framing Label by Gender")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, plots_dir / "q6_framing.png")


def _plot_score_distributions(df: pd.DataFrame, plots_dir: Path) -> None:
    """Box plots: score distributions per gender (base condition)."""
    base = df[df["prompt_type"] == "base"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, col in zip(axes, CONTINUOUS):
        valid = base[base[col] != 999]
        if valid.empty:
            continue
        sns.boxplot(data=valid, x="gender", y=col, ax=ax, palette="muted")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.set_ylim(0, 10)
    fig.suptitle("Score Distributions by Gender (base condition)", fontsize=11, y=1.01)
    _save(fig, plots_dir / "score_distributions.png")


def _plot_intersectionality(df: pd.DataFrame, plots_dir: Path) -> None:
    """Interaction plots: gender × age per continuous outcome (base condition)."""
    base = df[df["prompt_type"] == "base"]
    for col in CONTINUOUS:
        valid = base[base[col] != 999]
        if valid.empty:
            continue
        grp  = valid.groupby(["gender", "age_group"])[col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        for age, subset in grp.groupby("age_group"):
            ax.plot(subset["gender"], subset[col], marker="o", label=str(age))
        ax.set_xlabel("Gender")
        ax.set_ylabel(f"Mean {col.replace('_', ' ')}")
        ax.set_title(f"Intersectionality: {col.replace('_', ' ').title()} (gender × age)")
        ax.legend(title="Age group", fontsize=8)
        ax.set_ylim(0, 10)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        _save(fig, plots_dir / f"intersectionality_{col}.png")


def _plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate all visualizations from requirements.md § Visualizations."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_scores_by_group(df, "gender", plots_dir)
    _plot_scores_by_group(df, "age_group", plots_dir)
    _plot_base_vs_mitigation(df, plots_dir)
    _plot_neutral_vs_base(df, plots_dir)
    _plot_q1_diagnosis_rate(df, plots_dir)
    _plot_q4b_treatment_type(df, plots_dir)
    _plot_q6_framing(df, plots_dir)
    _plot_score_distributions(df, plots_dir)
    _plot_intersectionality(df, plots_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(
    results_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Load results.csv, run the full analysis pipeline, write all output files.

    Parameters
    ----------
    results_path : path to results.csv (default: data/processed/results.csv)
    output_dir   : root for output files (default: results/)
    """
    csv_path = results_path or (_PROCESSED_DIR / "results.csv")
    out_dir  = output_dir   or _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analysis: loading {csv_path}")
    df = _load(csv_path)
    print(f"  {len(df)} rows loaded")

    print("Analysis: descriptive statistics")
    agg = _descriptive(df)
    agg_path = out_dir / "aggregated_results.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  Written {agg_path.name}  ({len(agg)} rows)")

    print("Analysis: inferential statistics")
    stat = _inferential(df)
    stat_path = out_dir / "statistical_results.csv"
    stat.to_csv(stat_path, index=False)
    print(f"  Written {stat_path.name}  ({len(stat)} rows)")

    print("Analysis: generating plots")
    _plots(df, out_dir / "plots")

    print("Analysis complete.")
