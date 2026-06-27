"""
src/analysis.py
Implements: design.md § Analysis Methods, requirements.md § Analysis.
Descriptive statistics only — no scipy, no pingouin, no inferential tests.
Reads data/processed/results_*.csv partitions, concatenates, runs analysis,
writes results/aggregated_results.csv, results/comparisons_{gender,age,intersectionality}.csv,
and results/plots/*.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
_RESULTS_DIR   = Path(__file__).parent.parent / "results"
_PLOTS_DIR     = _RESULTS_DIR / "plots"

_CONTINUOUS     = ['severity_score', 'impairment_score', 'treatment_score', 'minimizing_score']
_CATEGORICAL    = ['diagnosis_yes_no', 'treatment_type_profile', 'framing_label']
_GENDER_ORDER   = ['woman', 'man', 'non-binary', 'trans-woman', 'trans-man']
_AGE_ORDER      = ['18-25', '40-55', '65+']
_SEVERITY_ORDER = ['ambiguous', 'mild', 'moderate', 'severe']

# Fixed colors for Q6 framing labels — consistent across all framing plots
_FRAMING_COLORS = {
    'gendered':    '#e07b54',   # orange-red
    'medicalized': '#5b8db8',   # steel blue
    'distanced':   '#6aab6a',   # muted green
    'neutral':     '#a882c6',   # soft purple
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    partitions = sorted(_PROCESSED_DIR.glob('results_*.csv'))
    if not partitions:
        combined = _PROCESSED_DIR / 'results.csv'
        if combined.exists():
            partitions = [combined]
        else:
            print("  ERROR: no results CSV found in data/processed/",
                  file=sys.stderr, flush=True)
            return pd.DataFrame()

    frames = [pd.read_csv(p) for p in partitions]
    df = pd.concat(frames, ignore_index=True)

    combined_path = _PROCESSED_DIR / 'results.csv'
    df.to_csv(combined_path, index=False)

    df = df[df['diagnosis_yes_no'] != 999].copy()
    return df


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> writing {path}", flush=True)
    df.to_csv(path, index=False)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> writing {path}", flush=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── descriptive stats helpers ─────────────────────────────────────────────────

def _desc(df: pd.DataFrame, group_cols: list[str], outcome: str) -> pd.DataFrame:
    """Mean, SD, min, max, range, N for a continuous outcome by group."""
    sub = df[df[outcome] != 999].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors='coerce')
    g = sub.groupby(group_cols)[outcome]
    return g.agg(
        mean='mean', std='std', min='min', max='max', N='count'
    ).assign(range=lambda r: r['max'] - r['min']).reset_index()


def _proportions(df: pd.DataFrame, group_cols: list[str], cat_col: str) -> pd.DataFrame:
    """Frequency counts and proportions per category per group."""
    sub = df[df[cat_col].notna() & (df[cat_col].astype(str) != '999')].copy()
    counts = sub.groupby(group_cols + [cat_col]).size().rename('count').reset_index()
    totals = sub.groupby(group_cols).size().rename('total').reset_index()
    merged = counts.merge(totals, on=group_cols)
    merged['proportion'] = merged['count'] / merged['total']
    return merged


def _valid_severities(df: pd.DataFrame) -> list[str]:
    """Return severity levels present in df, in canonical order."""
    return [s for s in _SEVERITY_ORDER if s in df['severity_level'].unique()]


def _grid_axes(n_items: int, n_cols: int = 2, subplot_w: int = 7, subplot_h: int = 5):
    """Create a 2-column grid of subplots and return (fig, flat axes list)."""
    n_rows = (n_items + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(subplot_w * n_cols, subplot_h * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten().tolist()
    return fig, axes_flat


def _hide_unused(axes_flat: list, n_used: int) -> None:
    for ax in axes_flat[n_used:]:
        ax.set_visible(False)


# ── aggregated_results.csv ────────────────────────────────────────────────────

def _aggregated(df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for out in _CONTINUOUS:
        r = _desc(df[df['prompt_type'] == 'base'], ['model', 'gender', 'severity_level'], out)
        r['outcome'] = out
        r['dimension'] = 'gender'
        frames.append(r.rename(columns={'gender': 'group_value'}))

    for out in _CONTINUOUS:
        r = _desc(df[df['prompt_type'] == 'base'], ['model', 'age_group', 'severity_level'], out)
        r['outcome'] = out
        r['dimension'] = 'age_group'
        frames.append(r.rename(columns={'age_group': 'group_value'}))

    r = _proportions(df[df['prompt_type'] == 'base'],
                     ['model', 'gender', 'severity_level'], 'diagnosis_yes_no')
    r['outcome'] = 'diagnosis_yes_no'
    r['dimension'] = 'gender'
    frames.append(r.rename(columns={'gender': 'group_value',
                                    'diagnosis_yes_no': 'category'}))

    r = _proportions(df[df['prompt_type'] == 'base'],
                     ['model', 'gender', 'severity_level'], 'treatment_type_profile')
    r['outcome'] = 'treatment_type_profile'
    r['dimension'] = 'gender'
    frames.append(r.rename(columns={'gender': 'group_value',
                                    'treatment_type_profile': 'category'}))

    r = _proportions(df[df['prompt_type'] == 'base'],
                     ['model', 'gender', 'severity_level'], 'framing_label')
    r['outcome'] = 'framing_label'
    r['dimension'] = 'gender'
    frames.append(r.rename(columns={'gender': 'group_value',
                                    'framing_label': 'category'}))

    return pd.concat(frames, ignore_index=True)


# ── comparisons (three files) ─────────────────────────────────────────────────

def _mean_for(cond_df: pd.DataFrame, out: str) -> float:
    sub = cond_df[cond_df[out] != 999].copy()
    sub[out] = pd.to_numeric(sub[out], errors='coerce')
    return sub[out].mean()


def _pooled_sd(rows_a: pd.DataFrame, rows_b: pd.DataFrame, out: str) -> float:
    sd_a = pd.to_numeric(rows_a[rows_a[out] != 999][out], errors='coerce').std()
    sd_b = pd.to_numeric(rows_b[rows_b[out] != 999][out], errors='coerce').std()
    if pd.notna(sd_a) and pd.notna(sd_b):
        return ((sd_a ** 2 + sd_b ** 2) / 2) ** 0.5
    return float('nan')


def _std_diff(effect: float, pooled: float) -> float:
    if pd.notna(effect) and pooled and pooled > 0:
        return effect / pooled
    return float('nan')


def _diff(a: float, b: float) -> float:
    return a - b if pd.notna(a) and pd.notna(b) else float('nan')


def _comparisons_gender(df: pd.DataFrame) -> pd.DataFrame:
    """160 rows: model × severity × outcome × gender (averaged over age groups).
    neutral_gender_mean is averaged over all three age groups (no specific age to anchor to).
    neutral_age_mean is the gender-specific, age-absent baseline (N=10 per cell).
    gender_effect = base(g) − neutral_gender_avg  (adding gender label vs no gender)
    age_effect    = base(g) − neutral_age(g)       (adding age label vs no age, for this gender)
    """
    frames = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        for sev in df['severity_level'].unique():
            sdf = mdf[mdf['severity_level'] == sev]
            for out in _CONTINUOUS:
                nf_rows = sdf[sdf['prompt_type'] == 'neutral_full']
                nf_mean = _mean_for(nf_rows, out)
                # neutral_gender averaged over all ages (no age dimension in this file)
                ng_mean = _mean_for(sdf[sdf['prompt_type'] == 'neutral_gender'], out)

                for g in _GENDER_ORDER:
                    base_rows = sdf[(sdf['prompt_type'] == 'base') & (sdf['gender'] == g)]
                    mit_rows  = sdf[(sdf['prompt_type'] == 'mitigation') & (sdf['gender'] == g)]
                    na_rows   = sdf[(sdf['prompt_type'] == 'neutral_age') & (sdf['gender'] == g)]

                    base_mean = _mean_for(base_rows, out)
                    mit_mean  = _mean_for(mit_rows, out)
                    na_mean   = _mean_for(na_rows, out)

                    total = _diff(base_mean, nf_mean)
                    psd   = _pooled_sd(base_rows, nf_rows, out)

                    frames.append({
                        'model': model, 'severity_level': sev,
                        'outcome': out, 'gender': g,
                        'base_mean': base_mean, 'mitigation_mean': mit_mean,
                        'neutral_full_mean': nf_mean,
                        'neutral_age_mean': na_mean,
                        'neutral_gender_mean': ng_mean,
                        'total_effect_base_minus_neutral_full': total,
                        'gender_effect_base_minus_neutral_gender': _diff(base_mean, ng_mean),
                        'age_effect_base_minus_neutral_age': _diff(base_mean, na_mean),
                        'mitigation_effect_base_minus_mitigation': _diff(base_mean, mit_mean),
                        'descriptive_std_diff': _std_diff(total, psd),
                    })
    return pd.DataFrame(frames)


def _comparisons_age(df: pd.DataFrame) -> pd.DataFrame:
    """96 rows: model × severity × outcome × age_group (averaged over genders).
    neutral_age_mean is averaged over all five genders (no specific gender to anchor to).
    neutral_gender_mean is the age-specific, gender-absent baseline (N=10 per cell).
    gender_effect = base(a) − neutral_gender(a)    (adding gender label vs no gender, for this age)
    age_effect    = base(a) − neutral_age_avg       (adding age label vs no age)
    """
    frames = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        for sev in df['severity_level'].unique():
            sdf = mdf[mdf['severity_level'] == sev]
            for out in _CONTINUOUS:
                nf_rows = sdf[sdf['prompt_type'] == 'neutral_full']
                nf_mean = _mean_for(nf_rows, out)
                # neutral_age averaged over all genders (no gender dimension in this file)
                na_mean_avg = _mean_for(sdf[sdf['prompt_type'] == 'neutral_age'], out)

                for age in _AGE_ORDER:
                    base_rows = sdf[(sdf['prompt_type'] == 'base') &
                                    (sdf['age_group'] == age) &
                                    sdf['gender'].isin(_GENDER_ORDER)]
                    mit_rows  = sdf[(sdf['prompt_type'] == 'mitigation') &
                                    (sdf['age_group'] == age) &
                                    sdf['gender'].isin(_GENDER_ORDER)]
                    ng_rows   = sdf[(sdf['prompt_type'] == 'neutral_gender') &
                                    (sdf['age_group'] == age)]

                    base_mean = _mean_for(base_rows, out)
                    mit_mean  = _mean_for(mit_rows, out)
                    ng_mean   = _mean_for(ng_rows, out)

                    total = _diff(base_mean, nf_mean)
                    psd   = _pooled_sd(base_rows, nf_rows, out)

                    frames.append({
                        'model': model, 'severity_level': sev,
                        'outcome': out, 'age_group': age,
                        'base_mean': base_mean, 'mitigation_mean': mit_mean,
                        'neutral_full_mean': nf_mean,
                        'neutral_age_mean': na_mean_avg,
                        'neutral_gender_mean': ng_mean,
                        'total_effect_base_minus_neutral_full': total,
                        'gender_effect_base_minus_neutral_gender': _diff(base_mean, ng_mean),
                        'age_effect_base_minus_neutral_age': _diff(base_mean, na_mean_avg),
                        'mitigation_effect_base_minus_mitigation': _diff(base_mean, mit_mean),
                        'descriptive_std_diff': _std_diff(total, psd),
                    })
    return pd.DataFrame(frames)


def _comparisons_intersectionality(df: pd.DataFrame) -> pd.DataFrame:
    """480 rows: model × severity × outcome × gender × age_group.
    Each cell uses the specific gender+age base mean.
    neutral_age_mean(g):   gender g, no age  — varies by gender, same across ages
    neutral_gender_mean(a): age a, no gender — varies by age, same across genders
    neutral_full_mean:      no demographics  — same for all cells
    gender_effect = base(g,a) − neutral_gender(a)  (adding gender when age is fixed)
    age_effect    = base(g,a) − neutral_age(g)      (adding age when gender is fixed)
    """
    frames = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        for sev in df['severity_level'].unique():
            sdf = mdf[mdf['severity_level'] == sev]
            for out in _CONTINUOUS:
                nf_rows = sdf[sdf['prompt_type'] == 'neutral_full']
                nf_mean = _mean_for(nf_rows, out)

                for g in _GENDER_ORDER:
                    na_mean = _mean_for(
                        sdf[(sdf['prompt_type'] == 'neutral_age') & (sdf['gender'] == g)], out)

                    for age in _AGE_ORDER:
                        base_rows = sdf[(sdf['prompt_type'] == 'base') &
                                        (sdf['gender'] == g) & (sdf['age_group'] == age)]
                        mit_rows  = sdf[(sdf['prompt_type'] == 'mitigation') &
                                        (sdf['gender'] == g) & (sdf['age_group'] == age)]
                        ng_mean   = _mean_for(
                            sdf[(sdf['prompt_type'] == 'neutral_gender') &
                                (sdf['age_group'] == age)], out)

                        base_mean = _mean_for(base_rows, out)
                        mit_mean  = _mean_for(mit_rows, out)

                        total = _diff(base_mean, nf_mean)
                        psd   = _pooled_sd(base_rows, nf_rows, out)

                        frames.append({
                            'model': model, 'severity_level': sev,
                            'outcome': out, 'gender': g, 'age_group': age,
                            'base_mean': base_mean, 'mitigation_mean': mit_mean,
                            'neutral_full_mean': nf_mean,
                            'neutral_age_mean': na_mean,
                            'neutral_gender_mean': ng_mean,
                            'total_effect_base_minus_neutral_full': total,
                            'gender_effect_base_minus_neutral_gender': _diff(base_mean, ng_mean),
                            'age_effect_base_minus_neutral_age': _diff(base_mean, na_mean),
                            'mitigation_effect_base_minus_mitigation': _diff(base_mean, mit_mean),
                            'descriptive_std_diff': _std_diff(total, psd),
                        })
    return pd.DataFrame(frames)


# ── plots ─────────────────────────────────────────────────────────────────────



def _stacked_bar(sdf: pd.DataFrame, col: str, ax,
                 group_col: str = 'gender',
                 group_order: list | None = None,
                 title: str | None = None,
                 col_colors: dict | None = None) -> bool:
    """Draw a stacked bar of `col` proportions by `group_col` onto `ax`.
    group_order: display order for rows (defaults to _GENDER_ORDER).
    col_colors: optional {label: hex} dict — fixes column order and colors.
    Returns False if no data; the caller should hide the axis in that case."""
    if group_order is None:
        group_order = _GENDER_ORDER
    valid = sdf[sdf[col].astype(str) != '999']
    if valid.empty:
        return False
    counts = valid.groupby([group_col, col]).size().reset_index(name='count')
    totals = valid.groupby(group_col).size().reset_index(name='total')
    merged = counts.merge(totals, on=group_col)
    merged['proportion'] = merged['count'] / merged['total']
    pivot = merged.pivot(index=group_col, columns=col,
                         values='proportion').fillna(0)
    rows = [g for g in group_order if g in pivot.index]
    if not rows:
        return False

    if col_colors:
        for label in col_colors:
            if label not in pivot.columns:
                pivot[label] = 0.0
        ordered_cols = [c for c in col_colors if c in pivot.columns]
        colors = [col_colors[c] for c in ordered_cols]
        pivot.loc[rows, ordered_cols].plot(kind='bar', stacked=True, ax=ax,
                                           rot=30, color=colors)
    else:
        pivot.loc[rows].plot(kind='bar', stacked=True, ax=ax, rot=30)

    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    if title:
        ax.set_title(title)
    return True




def _plot_framing(df: pd.DataFrame, plots_dir: Path) -> None:
    """Stacked bar: framing label proportions by gender and by age group.
    Produces four files per model: gender grid, gender combined, age grid, age combined."""
    base = df[(df['prompt_type'] == 'base') &
              df['gender'].isin(_GENDER_ORDER) &
              df['age_group'].isin(_AGE_ORDER)].copy()
    models     = sorted(df['model'].unique())
    severities = _valid_severities(base)

    for model in models:
        mdf = base[base['model'] == model]
        slug = model.replace(':', '_')

        # ── by gender ────────────────────────────────────────────────────────
        fig, axes_flat = _grid_axes(len(severities), n_cols=2, subplot_w=7, subplot_h=5)
        for i, sev in enumerate(severities):
            ax = axes_flat[i]
            ok = _stacked_bar(mdf[mdf['severity_level'] == sev], 'framing_label', ax,
                              group_col='gender', group_order=_GENDER_ORDER,
                              title=sev, col_colors=_FRAMING_COLORS)
            if not ok:
                ax.set_visible(False)
        _hide_unused(axes_flat, len(severities))
        fig.suptitle(f"Q6 framing label by gender — {model} (base condition)", fontsize=13)
        fig.tight_layout()
        _save_fig(fig, plots_dir / f"q6_framing_{slug}.png")

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        _stacked_bar(mdf, 'framing_label', ax2, group_col='gender',
                     group_order=_GENDER_ORDER, col_colors=_FRAMING_COLORS)
        fig2.suptitle(f"Q6 framing label by gender — {model} (base, all severities combined)",
                      fontsize=12)
        fig2.tight_layout()
        _save_fig(fig2, plots_dir / f"q6_framing_combined_{slug}.png")

        # ── by age group ──────────────────────────────────────────────────────
        fig3, axes_flat3 = _grid_axes(len(severities), n_cols=2, subplot_w=7, subplot_h=5)
        for i, sev in enumerate(severities):
            ax = axes_flat3[i]
            ok = _stacked_bar(mdf[mdf['severity_level'] == sev], 'framing_label', ax,
                              group_col='age_group', group_order=_AGE_ORDER,
                              title=sev, col_colors=_FRAMING_COLORS)
            if not ok:
                ax.set_visible(False)
        _hide_unused(axes_flat3, len(severities))
        fig3.suptitle(f"Q6 framing label by age group — {model} (base condition)", fontsize=13)
        fig3.tight_layout()
        _save_fig(fig3, plots_dir / f"q6_framing_age_{slug}.png")

        fig4, ax4 = plt.subplots(figsize=(9, 5))
        _stacked_bar(mdf, 'framing_label', ax4, group_col='age_group',
                     group_order=_AGE_ORDER, col_colors=_FRAMING_COLORS)
        fig4.suptitle(f"Q6 framing label by age group — {model} (base, all severities combined)",
                      fontsize=12)
        fig4.tight_layout()
        _save_fig(fig4, plots_dir / f"q6_framing_age_combined_{slug}.png")


_GENDER_MARKERS = {'man': 'o', 'woman': 's', 'non-binary': '^',
                   'trans-woman': 'D', 'trans-man': 'v'}

_GENDER_LINE_COLORS = {
    'woman':      '#e07b54',
    'man':        '#5b8db8',
    'non-binary': '#6aab6a',
    'trans-woman':'#a882c6',
    'trans-man':  '#d4a843',
}


def _plot_framing_medicalization(df: pd.DataFrame, plots_dir: Path) -> None:
    """Line plot: % medicalized framing by gender × age (intersectionality focus).
    One panel per model. Y-axis auto-scaled. Key finding: trans identities spike at 65+."""
    base = df[(df['prompt_type'] == 'base') &
              df['gender'].isin(_GENDER_ORDER) &
              df['age_group'].isin(_AGE_ORDER) &
              (df['framing_label'].astype(str) != '999')].copy()
    models = sorted(base['model'].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = base[base['model'] == model]
        counts = mdf.groupby(['gender', 'age_group', 'framing_label']).size().reset_index(name='n')
        totals = mdf.groupby(['gender', 'age_group']).size().reset_index(name='total')
        merged = counts.merge(totals, on=['gender', 'age_group'])
        merged['pct'] = merged['n'] / merged['total'] * 100
        med = merged[merged['framing_label'] == 'medicalized']

        all_vals = []
        for g in _GENDER_ORDER:
            gdf = med[med['gender'] == g].sort_values('age_group')
            gdf = gdf.set_index('age_group').reindex(_AGE_ORDER).reset_index()
            marker = _GENDER_MARKERS.get(g, 'o')
            color  = _GENDER_LINE_COLORS.get(g, None)
            ax.plot(gdf['age_group'], gdf['pct'], marker=marker, linewidth=2.5,
                    markersize=9, label=g, color=color)
            all_vals.extend(gdf['pct'].dropna().tolist())

        # y-axis: data range + padding, but always start at 0 if min > 0
        lo = max(0, min(all_vals) - 8)
        hi = min(100, max(all_vals) + 8)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.set_xlabel('Age group')
        ax.set_ylabel('% medicalized framing')
        ax.set_title(model, fontsize=11)
        ax.legend(title='Gender', fontsize=9)
        ax.grid(axis='y', alpha=0.4)

    fig.suptitle(
        'Q6 framing — medicalized label by gender × age  (base condition, all severities)\n'
        'Rising medicalization with age is strongest for trans identities',
        fontsize=12)
    fig.tight_layout()
    _save_fig(fig, plots_dir / 'q6_framing_medicalization_intersectionality.png')


_AGE_COLORS = {'18-25': '#5b8db8', '40-55': '#6aab6a', '65+': '#e07b54'}


def _plot_diagnosis_ambiguous(df: pd.DataFrame, plots_dir: Path) -> None:
    """Three plots for Q1 diagnosis rate, ambiguous vignette only, base condition.
    1. By gender  2. By age  3. Intersectionality (gender × age, line plot).
    Uses _GENDER_LINE_COLORS / _AGE_COLORS for consistency with Q6 palette."""
    amb = df[(df['severity_level'] == 'ambiguous') &
             (df['prompt_type'] == 'base') &
             df['gender'].isin(_GENDER_ORDER) &
             df['age_group'].isin(_AGE_ORDER)].copy()
    models = sorted(amb['model'].unique())

    def diag_rate(grp_df, by):
        g = grp_df.groupby(by)['diagnosis_yes_no'].agg(['sum', 'count'])
        g['rate'] = g['sum'] / g['count'] * 100
        return g['rate']

    # ── 1. By gender ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        rates = diag_rate(amb[amb['model'] == model], 'gender').reindex(_GENDER_ORDER)
        colors = [_GENDER_LINE_COLORS.get(g, '#999999') for g in _GENDER_ORDER]
        bars = ax.bar(_GENDER_ORDER, rates, color=colors, edgecolor='white', width=0.6)
        for bar, val in zip(bars, rates):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
        ax.set_title(model, fontsize=11)
        ax.set_ylim(0, 110)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=20)
        if ax == axes[0]:
            ax.set_ylabel('% diagnosed depression')
    fig.suptitle('Q1 diagnosis rate by gender — ambiguous vignette (base condition)', fontsize=12)
    fig.tight_layout()
    _save_fig(fig, plots_dir / 'q1_diagnosis_ambiguous_gender.png')

    # ── 2. By age ─────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes2 = [axes2]
    for ax, model in zip(axes2, models):
        rates = diag_rate(amb[amb['model'] == model], 'age_group').reindex(_AGE_ORDER)
        colors = [_AGE_COLORS.get(a, '#999999') for a in _AGE_ORDER]
        bars = ax.bar(_AGE_ORDER, rates, color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, rates):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
        ax.set_title(model, fontsize=11)
        ax.set_ylim(0, 110)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.set_xlabel('Age group')
        if ax == axes2[0]:
            ax.set_ylabel('% diagnosed depression')
    fig2.suptitle('Q1 diagnosis rate by age group — ambiguous vignette (base condition)', fontsize=12)
    fig2.tight_layout()
    _save_fig(fig2, plots_dir / 'q1_diagnosis_ambiguous_age.png')

    # ── 3. Intersectionality: line plot gender × age ───────────────────────────
    fig3, axes3 = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes3 = [axes3]
    for ax, model in zip(axes3, models):
        mdf = amb[amb['model'] == model]
        all_vals = []
        for g in _GENDER_ORDER:
            rates = (mdf[mdf['gender'] == g]
                     .groupby('age_group')['diagnosis_yes_no']
                     .agg(['sum', 'count']))
            rates['rate'] = rates['sum'] / rates['count'] * 100
            rates = rates['rate'].reindex(_AGE_ORDER)
            color  = _GENDER_LINE_COLORS.get(g, '#999999')
            marker = _GENDER_MARKERS.get(g, 'o')
            ax.plot(_AGE_ORDER, rates, marker=marker, linewidth=2.5,
                    markersize=9, label=g, color=color)
            all_vals.extend(rates.dropna().tolist())
        lo = max(0, min(all_vals) - 10)
        hi = min(100, max(all_vals) + 10)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.set_xlabel('Age group')
        ax.set_title(model, fontsize=11)
        if ax == axes3[0]:
            ax.set_ylabel('% diagnosed depression')
        ax.legend(title='Gender', fontsize=9)
        ax.grid(axis='y', alpha=0.4)
    fig3.suptitle('Q1 diagnosis rate by gender × age — ambiguous vignette (base condition)',
                  fontsize=12)
    fig3.tight_layout()
    _save_fig(fig3, plots_dir / 'q1_diagnosis_ambiguous_intersectionality.png')


def _plot_diagnosis_mitigation(df: pd.DataFrame, plots_dir: Path) -> None:
    """Base vs mitigation comparison for Q1 diagnosis rate (ambiguous vignette only).
    Produces three plots: gender grouped bars, age grouped bars, intersectionality heatmaps."""
    amb = df[(df['severity_level'] == 'ambiguous') &
             df['prompt_type'].isin(['base', 'mitigation']) &
             df['gender'].isin(_GENDER_ORDER) &
             df['age_group'].isin(_AGE_ORDER) &
             (df['diagnosis_yes_no'] != 999)].copy()
    models = sorted(amb['model'].unique())

    def rates_by(mdf, group_col, order):
        out = {}
        for pt in ['base', 'mitigation']:
            g = mdf[mdf['prompt_type'] == pt].groupby(group_col)['diagnosis_yes_no']
            out[pt] = (g.sum() / g.count() * 100).reindex(order)
        return out['base'], out['mitigation']

    # ── 1. Gender: grouped bars (base solid, mitigation hatched) ─────────────
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    x = range(len(_GENDER_ORDER))
    w = 0.38
    for ax, model in zip(axes, models):
        base_r, mit_r = rates_by(amb[amb['model'] == model], 'gender', _GENDER_ORDER)
        colors = [_GENDER_LINE_COLORS.get(g, '#999') for g in _GENDER_ORDER]
        bars_b = ax.bar([i - w / 2 for i in x], base_r, w, color=colors, label='base')
        bars_m = ax.bar([i + w / 2 for i in x], mit_r, w, color=colors, alpha=0.45,
                        hatch='//', edgecolor='white', label='mitigation')
        for bars, vals in [(bars_b, base_r), (bars_m, mit_r)]:
            for bar, val in zip(bars, vals):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(_GENDER_ORDER, rotation=20, ha='right')
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.set_title(model, fontsize=11)
        ax.legend(fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('% diagnosed depression')
    fig.suptitle('Q1 diagnosis rate — base vs mitigation by gender (ambiguous vignette)', fontsize=12)
    fig.tight_layout()
    _save_fig(fig, plots_dir / 'q1_diagnosis_mitigation_gender.png')

    # ── 2. Age: grouped bars ─────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes2 = [axes2]
    xa = range(len(_AGE_ORDER))
    for ax, model in zip(axes2, models):
        base_r, mit_r = rates_by(amb[amb['model'] == model], 'age_group', _AGE_ORDER)
        colors = [_AGE_COLORS.get(a, '#999') for a in _AGE_ORDER]
        bars_b = ax.bar([i - w / 2 for i in xa], base_r, w, color=colors, label='base')
        bars_m = ax.bar([i + w / 2 for i in xa], mit_r, w, color=colors, alpha=0.45,
                        hatch='//', edgecolor='white', label='mitigation')
        for bars, vals in [(bars_b, base_r), (bars_m, mit_r)]:
            for bar, val in zip(bars, vals):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(list(xa))
        ax.set_xticklabels(_AGE_ORDER)
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.set_title(model, fontsize=11)
        ax.legend(fontsize=9)
        if ax == axes2[0]:
            ax.set_ylabel('% diagnosed depression')
    fig2.suptitle('Q1 diagnosis rate — base vs mitigation by age group (ambiguous vignette)', fontsize=12)
    fig2.tight_layout()
    _save_fig(fig2, plots_dir / 'q1_diagnosis_mitigation_age.png')

    # ── 3. Intersectionality: change heatmaps (mitigation − base) ────────────
    fig3, axes3 = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes3 = [axes3]
    for ax, model in zip(axes3, models):
        mdf = amb[amb['model'] == model]
        base_p = (mdf[mdf['prompt_type'] == 'base']
                  .groupby(['gender', 'age_group'])['diagnosis_yes_no'].mean() * 100)
        mit_p  = (mdf[mdf['prompt_type'] == 'mitigation']
                  .groupby(['gender', 'age_group'])['diagnosis_yes_no'].mean() * 100)
        diff = (mit_p - base_p).unstack('age_group').reindex(_GENDER_ORDER).reindex(
            columns=_AGE_ORDER).fillna(0)
        im = ax.imshow(diff.values, cmap='RdBu', aspect='auto', vmin=-60, vmax=60)
        ax.set_xticks(range(len(_AGE_ORDER)))
        ax.set_yticks(range(len(_GENDER_ORDER)))
        ax.set_xticklabels(_AGE_ORDER)
        ax.set_yticklabels(_GENDER_ORDER)
        ax.set_xlabel('Age group')
        ax.set_title(model, fontsize=11)
        for row in range(len(_GENDER_ORDER)):
            for col in range(len(_AGE_ORDER)):
                val = diff.iloc[row, col]
                ax.text(col, row, f'{val:+.0f}pp', ha='center', va='center',
                        color='white' if abs(val) > 35 else 'black', fontsize=10, fontweight='bold')
        fig3.colorbar(im, ax=ax, label='pp change (mitigation − base)')
    if len(axes3) > 0:
        axes3[0].set_ylabel('Gender')
    fig3.suptitle('Q1 diagnosis rate change: mitigation − base (ambiguous vignette)', fontsize=12)
    fig3.tight_layout()
    _save_fig(fig3, plots_dir / 'q1_diagnosis_mitigation_intersectionality.png')




# ── main entry point ──────────────────────────────────────────────────────────

def _plot_score_effects(df: pd.DataFrame, plots_dir: Path, outcome: str) -> None:
    """Three plots for any continuous outcome — gender bars, age bars, intersectionality lines.
    All use auto-scaled y-axes so small differences are visible.
    Grid layout: rows = severity levels, cols = models.
    Color palette: _GENDER_LINE_COLORS for gender, _AGE_COLORS for age."""
    label = outcome.replace('_', ' ')
    base = df[(df['prompt_type'] == 'base') &
              df['gender'].isin(_GENDER_ORDER) &
              df['age_group'].isin(_AGE_ORDER) &
              (df[outcome] != 999)].copy()
    base[outcome] = pd.to_numeric(base[outcome], errors='coerce')
    models     = sorted(base['model'].unique())
    severities = _valid_severities(base)
    n_sev, n_mod = len(severities), len(models)

    # ── 1. Gender bars — n_sev × n_mod grid ──────────────────────────────────
    fig, axes = plt.subplots(n_sev, n_mod,
                             figsize=(7 * n_mod, 4 * n_sev), squeeze=False)
    for row, sev in enumerate(severities):
        sdf = base[base['severity_level'] == sev]
        for col, model in enumerate(models):
            ax = axes[row][col]
            mdf = sdf[sdf['model'] == model]
            means = mdf.groupby('gender')[outcome].mean().reindex(_GENDER_ORDER)
            sds   = mdf.groupby('gender')[outcome].std().reindex(_GENDER_ORDER)
            colors = [_GENDER_LINE_COLORS.get(g, '#999') for g in _GENDER_ORDER]
            bars = ax.bar(_GENDER_ORDER, means, yerr=sds, color=colors,
                          edgecolor='white', width=0.6,
                          error_kw={'elinewidth': 1.2, 'capsize': 4, 'alpha': 0.6})
            for bar, val in zip(bars, means):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (sds.max() or 0) + 0.05,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)
            lo = max(0, means.min() - (sds.max() or 0) - 0.4)
            hi = min(10, means.max() + (sds.max() or 0) + 0.5)
            ax.set_ylim(lo, hi)
            ax.set_title(f'{sev}  |  {model}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel(f'Mean {label}' if col == 0 else '')
            ax.tick_params(axis='x', rotation=25)
    fig.suptitle(f'{label.title()} by gender — base condition  (bars = ±1 SD)', fontsize=13)
    fig.tight_layout()
    _save_fig(fig, plots_dir / f'{outcome}_by_gender_per_severity.png')

    # ── 2. Age bars — n_sev × n_mod grid ─────────────────────────────────────
    fig2, axes2 = plt.subplots(n_sev, n_mod,
                               figsize=(6 * n_mod, 4 * n_sev), squeeze=False)
    for row, sev in enumerate(severities):
        sdf = base[base['severity_level'] == sev]
        for col, model in enumerate(models):
            ax = axes2[row][col]
            mdf = sdf[sdf['model'] == model]
            means = mdf.groupby('age_group')[outcome].mean().reindex(_AGE_ORDER)
            sds   = mdf.groupby('age_group')[outcome].std().reindex(_AGE_ORDER)
            colors = [_AGE_COLORS.get(a, '#999') for a in _AGE_ORDER]
            bars = ax.bar(_AGE_ORDER, means, yerr=sds, color=colors,
                          edgecolor='white', width=0.5,
                          error_kw={'elinewidth': 1.2, 'capsize': 4, 'alpha': 0.6})
            for bar, val in zip(bars, means):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (sds.max() or 0) + 0.05,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            lo = max(0, means.min() - (sds.max() or 0) - 0.4)
            hi = min(10, means.max() + (sds.max() or 0) + 0.5)
            ax.set_ylim(lo, hi)
            ax.set_title(f'{sev}  |  {model}', fontsize=10)
            ax.set_xlabel('Age group')
            ax.set_ylabel(f'Mean {label}' if col == 0 else '')
    fig2.suptitle(f'{label.title()} by age group — base condition  (bars = ±1 SD)', fontsize=13)
    fig2.tight_layout()
    _save_fig(fig2, plots_dir / f'{outcome}_by_age_per_severity.png')

    # ── 3. Intersectionality lines — n_sev × n_mod grid ──────────────────────
    fig3, axes3 = plt.subplots(n_sev, n_mod,
                               figsize=(7 * n_mod, 4 * n_sev), squeeze=False)
    for row, sev in enumerate(severities):
        sdf = base[base['severity_level'] == sev]
        for col, model in enumerate(models):
            ax = axes3[row][col]
            mdf = sdf[sdf['model'] == model]
            all_vals = []
            for g in _GENDER_ORDER:
                means = (mdf[mdf['gender'] == g]
                         .groupby('age_group')[outcome]
                         .mean().reindex(_AGE_ORDER))
                color  = _GENDER_LINE_COLORS.get(g, '#999')
                marker = _GENDER_MARKERS.get(g, 'o')
                ax.plot(_AGE_ORDER, means, marker=marker, linewidth=2,
                        markersize=7, label=g, color=color)
                all_vals.extend(means.dropna().tolist())
            if all_vals:
                pad = max((max(all_vals) - min(all_vals)) * 0.5, 0.15)
                ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
            ax.set_title(f'{sev}  |  {model}', fontsize=10)
            ax.set_xlabel('Age group')
            ax.set_ylabel(f'Mean {label}' if col == 0 else '')
            ax.grid(axis='y', alpha=0.35)
            if row == 0 and col == n_mod - 1:
                ax.legend(title='Gender', fontsize=8,
                          bbox_to_anchor=(1.01, 1), loc='upper left')
    fig3.suptitle(f'{label.title()} — gender × age intersectionality — base condition',
                  fontsize=13)
    fig3.tight_layout()
    _save_fig(fig3, plots_dir / f'{outcome}_intersectionality_per_severity.png')


_TREATMENT_COLORS = {
    'pharmacological': '#e07b54',
    'clinical':        '#5b8db8',
    'behavioral':      '#6aab6a',
}


def _plot_treatment_type_q4b(df: pd.DataFrame, plots_dir: Path) -> None:
    """Stacked bar plots (all three treatment types) by gender and age per severity,
    plus one overall pharmacological % heatmap (gender × age) per model."""
    base = df[(df['prompt_type'] == 'base') &
              df['gender'].isin(_GENDER_ORDER) &
              df['age_group'].isin(_AGE_ORDER) &
              (df['treatment_type_profile'].astype(str) != '999')].copy()
    models     = sorted(base['model'].unique())
    severities = _valid_severities(base)

    for model in models:
        mdf = base[base['model'] == model]
        slug = model.replace(':', '_')

        # ── stacked bars by gender, per severity grid ─────────────────────────
        fig, axes_flat = _grid_axes(len(severities), n_cols=2, subplot_w=7, subplot_h=5)
        for i, sev in enumerate(severities):
            ok = _stacked_bar(mdf[mdf['severity_level'] == sev],
                              'treatment_type_profile', axes_flat[i],
                              group_col='gender', group_order=_GENDER_ORDER,
                              title=sev, col_colors=_TREATMENT_COLORS)
            if not ok:
                axes_flat[i].set_visible(False)
        _hide_unused(axes_flat, len(severities))
        fig.suptitle(f"Q4b treatment type by gender — {model} (base condition)", fontsize=13)
        fig.tight_layout()
        _save_fig(fig, plots_dir / f"q4b_treatment_type_pharm_gender_{slug}.png")

        # ── stacked bars by gender, combined ─────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        _stacked_bar(mdf, 'treatment_type_profile', ax2,
                     group_col='gender', group_order=_GENDER_ORDER,
                     col_colors=_TREATMENT_COLORS)
        fig2.suptitle(f"Q4b treatment type by gender — {model} (base, all severities combined)", fontsize=12)
        fig2.tight_layout()
        _save_fig(fig2, plots_dir / f"q4b_treatment_type_pharm_gender_combined_{slug}.png")

        # ── stacked bars by age group, per severity grid ──────────────────────
        fig3, axes_flat3 = _grid_axes(len(severities), n_cols=2, subplot_w=7, subplot_h=5)
        for i, sev in enumerate(severities):
            ok = _stacked_bar(mdf[mdf['severity_level'] == sev],
                              'treatment_type_profile', axes_flat3[i],
                              group_col='age_group', group_order=_AGE_ORDER,
                              title=sev, col_colors=_TREATMENT_COLORS)
            if not ok:
                axes_flat3[i].set_visible(False)
        _hide_unused(axes_flat3, len(severities))
        fig3.suptitle(f"Q4b treatment type by age group — {model} (base condition)", fontsize=13)
        fig3.tight_layout()
        _save_fig(fig3, plots_dir / f"q4b_treatment_type_pharm_age_{slug}.png")

        # ── stacked bars by age group, combined ──────────────────────────────
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        _stacked_bar(mdf, 'treatment_type_profile', ax4,
                     group_col='age_group', group_order=_AGE_ORDER,
                     col_colors=_TREATMENT_COLORS)
        fig4.suptitle(f"Q4b treatment type by age group — {model} (base, all severities combined)", fontsize=12)
        fig4.tight_layout()
        _save_fig(fig4, plots_dir / f"q4b_treatment_type_pharm_age_combined_{slug}.png")

        # ── heatmap: overall pharmacological % (gender × age, all severities) ─
        counts = mdf.groupby(['gender', 'age_group', 'treatment_type_profile']).size().reset_index(name='n')
        totals = mdf.groupby(['gender', 'age_group']).size().reset_index(name='total')
        merged = counts.merge(totals, on=['gender', 'age_group'])
        merged['pct'] = merged['n'] / merged['total'] * 100
        pharm = merged[merged['treatment_type_profile'] == 'pharmacological'].copy()
        pivot = pharm.pivot(index='gender', columns='age_group', values='pct').fillna(0)
        pivot = pivot.reindex(_GENDER_ORDER).reindex(columns=_AGE_ORDER).fillna(0)

        fig5, ax5 = plt.subplots(figsize=(7, 5))
        im = ax5.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=60)
        ax5.set_xticks(range(len(_AGE_ORDER)))
        ax5.set_yticks(range(len(_GENDER_ORDER)))
        ax5.set_xticklabels(_AGE_ORDER)
        ax5.set_yticklabels(_GENDER_ORDER)
        ax5.set_xlabel('Age group')
        ax5.set_ylabel('Gender')
        for row in range(len(_GENDER_ORDER)):
            for col in range(len(_AGE_ORDER)):
                val = pivot.iloc[row, col]
                ax5.text(col, row, f'{val:.0f}%', ha='center', va='center',
                         color='white' if val > 35 else 'black', fontsize=11, fontweight='bold')
        fig5.colorbar(im, ax=ax5, label='% pharmacological')
        fig5.suptitle(f"Q4b pharmacological treatment % — gender × age — {model} (base, all severities)", fontsize=12)
        fig5.tight_layout()
        _save_fig(fig5, plots_dir / f"q4b_treatment_type_pharm_intersectionality_{slug}.png")


def _plot_mitigation_effects(df: pd.DataFrame, plots_dir: Path) -> None:
    """Three summary plots for base vs mitigation across Q2-Q6.
    1. Q4b pharmacological % by gender — overall + severe (2x2 grid)
    2. Q6 medicalized % by gender — overall + severe (2x2 grid)
    3. Continuous score change heatmap (mitigation-base) by gender, both models
    """
    import numpy as np

    bm = df[df['prompt_type'].isin(['base', 'mitigation']) &
            df['gender'].isin(_GENDER_ORDER) &
            df['age_group'].isin(_AGE_ORDER)].copy()
    models = ['llama3:8b', 'mistral:7b']
    panels = [('overall', None), ('severe', 'severe')]
    w = 0.38

    def _cat_pct(mdf, col, cat, sev=None):
        if sev:
            mdf = mdf[mdf['severity_level'] == sev]
        out = {}
        for pt in ['base', 'mitigation']:
            pdf = mdf[(mdf['prompt_type'] == pt) & (mdf[col].astype(str) != '999')]
            counts = pdf.groupby(['gender', col]).size().reset_index(name='n')
            totals = pdf.groupby('gender').size().reset_index(name='total')
            m = counts[counts[col] == cat].merge(totals, on='gender')
            m['pct'] = m['n'] / m['total'] * 100
            out[pt] = m.set_index('gender')['pct'].reindex(_GENDER_ORDER).fillna(0)
        return out['base'], out['mitigation']

    def _grouped_bars(ax, base_r, mit_r, title, ylabel, ylim):
        x = range(len(_GENDER_ORDER))
        colors = [_GENDER_LINE_COLORS.get(g, '#999') for g in _GENDER_ORDER]
        bars_b = ax.bar([i - w / 2 for i in x], base_r, w, color=colors, label='base')
        bars_m = ax.bar([i + w / 2 for i in x], mit_r, w, color=colors, alpha=0.45,
                        hatch='//', edgecolor='white', label='mitigation')
        for bars, vals in [(bars_b, base_r), (bars_m, mit_r)]:
            for bar, val in zip(bars, vals):
                if pd.notna(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(_GENDER_ORDER, rotation=20, ha='right')
        ax.set_ylim(0, ylim)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)

    # ── Plot 1: Q4b pharmacological % ────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    for ci, model in enumerate(models):
        mdf = bm[bm['model'] == model]
        slug = model.split(':')[0]
        for ri, (label, sev) in enumerate(panels):
            ax = axes1[ri, ci]
            base_r, mit_r = _cat_pct(mdf, 'treatment_type_profile', 'pharmacological', sev)
            ylim = 75 if sev else 35
            _grouped_bars(ax, base_r, mit_r, f'{slug} — {label}',
                          '% pharmacological', ylim)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=9)
    fig1.suptitle('Q4b pharmacological treatment % — base vs mitigation by gender', fontsize=13)
    fig1.tight_layout()
    _save_fig(fig1, plots_dir / 'mitigation_q4b_pharmacological_gender.png')

    # ── Plot 1b: Q4b clinical % ───────────────────────────────────────────────
    fig1b, axes1b = plt.subplots(2, 2, figsize=(14, 10))
    for ci, model in enumerate(models):
        mdf = bm[bm['model'] == model]
        slug = model.split(':')[0]
        for ri, (label, sev) in enumerate(panels):
            ax = axes1b[ri, ci]
            base_r, mit_r = _cat_pct(mdf, 'treatment_type_profile', 'clinical', sev)
            _grouped_bars(ax, base_r, mit_r, f'{slug} — {label}',
                          '% clinical', 110)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=9)
    fig1b.suptitle('Q4b clinical treatment % — base vs mitigation by gender', fontsize=13)
    fig1b.tight_layout()
    _save_fig(fig1b, plots_dir / 'mitigation_q4b_clinical_gender.png')

    # ── Plot 2: Q6 medicalized % ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    for ci, model in enumerate(models):
        mdf = bm[bm['model'] == model]
        slug = model.split(':')[0]
        for ri, (label, sev) in enumerate(panels):
            ax = axes2[ri, ci]
            base_r, mit_r = _cat_pct(mdf, 'framing_label', 'medicalized', sev)
            _grouped_bars(ax, base_r, mit_r, f'{slug} — {label}',
                          '% medicalized framing', 105)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=9)
    fig2.suptitle('Q6 medicalized framing % — base vs mitigation by gender', fontsize=13)
    fig2.tight_layout()
    _save_fig(fig2, plots_dir / 'mitigation_q6_medicalized_gender.png')

    # ── Plot 3: Continuous score change heatmap ───────────────────────────────
    cont = ['severity_score', 'impairment_score', 'treatment_score', 'minimizing_score']
    labels = ['severity\nscore', 'impairment\nscore', 'treatment\nscore', 'minimizing\nscore']
    for out in cont:
        bm[out] = pd.to_numeric(bm[out], errors='coerce')
        bm.loc[bm[out] == 999, out] = float('nan')

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model in zip(axes3, models):
        mdf = bm[bm['model'] == model]
        mat = np.array([
            [(mdf[(mdf['gender'] == g) & (mdf['prompt_type'] == 'mitigation')][out].mean() -
              mdf[(mdf['gender'] == g) & (mdf['prompt_type'] == 'base')][out].mean())
             for g in _GENDER_ORDER]
            for out in cont
        ])
        vmax = max(abs(mat).max(), 0.3)
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(_GENDER_ORDER)))
        ax.set_yticks(range(len(cont)))
        ax.set_xticklabels(_GENDER_ORDER, rotation=20, ha='right')
        ax.set_yticklabels(labels)
        ax.set_title(model, fontsize=11)
        for i in range(len(cont)):
            for j in range(len(_GENDER_ORDER)):
                val = mat[i, j]
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        color='white' if abs(val) > vmax * 0.6 else 'black', fontsize=9)
        fig3.colorbar(im, ax=ax, label='mitigation − base')
    fig3.suptitle('Continuous score changes: mitigation − base by gender (Q2–Q5)', fontsize=12)
    fig3.tight_layout()
    _save_fig(fig3, plots_dir / 'mitigation_continuous_change_gender.png')


def _plot_neutral_effects(df: pd.DataFrame, plots_dir: Path) -> None:
    """One 2×4 grouped-bar figure per outcome for gender AND age dimensions.

    Gender plots  (file: neutral_vs_base_<outcome>_gender.png)
      x = genders, solid = base, hatched = neutral_age (gender kept, age stripped)
      dashed line = neutral_full (no labels at all) — scalar per severity

    Age plots  (file: neutral_vs_base_<outcome>_age.png)
      x = age groups, solid = base (avg over genders), hatched = neutral_gender (age kept, gender stripped)
      dashed line = neutral_full (no gender, no age) — scalar per severity

    Each figure: 2 rows (models) × 4 cols (severities).
    """
    import numpy as np

    cont = ['severity_score', 'impairment_score', 'treatment_score', 'minimizing_score']

    # ── shared data prep ────────────────────────────────────────────────────
    base = df[df['prompt_type'] == 'base'].copy()
    neu_gender = df[df['prompt_type'] == 'neutral_gender'].copy()   # age kept, gender stripped
    neu_age    = df[df['prompt_type'] == 'neutral_age'].copy()       # gender kept, age stripped
    neu_full   = df[df['prompt_type'] == 'neutral_full'].copy()      # both stripped

    for c in cont:
        for d in (base, neu_gender, neu_age, neu_full):
            d[c] = pd.to_numeric(d[c], errors='coerce')
            d.loc[d[c] == 999, c] = float('nan')

    models = ['llama3:8b', 'mistral:7b']
    w = 0.38

    # ── outcome configs ──────────────────────────────────────────────────────
    OutCfg = [
        # (slug, display_label, col, cat_or_None)
        ('severity_score',   'severity score',       'severity_score',        None),
        ('impairment_score', 'impairment score',     'impairment_score',       None),
        ('treatment_score',  'treatment score',      'treatment_score',        None),
        ('minimizing_score', 'minimizing score',     'minimizing_score',       None),
        ('q4b_pharmacological', 'Q4b pharmacological %', 'treatment_type_profile', 'pharmacological'),
        ('q6_medicalized',   'Q6 medicalized %',    'framing_label',          'medicalized'),
    ]

    def _val(d, col, cat, gender_filter=None, age_filter=None):
        """Mean score or % for category cat, optionally filtered."""
        sub = d.copy()
        if gender_filter is not None:
            sub = sub[sub['gender'] == gender_filter]
        if age_filter is not None:
            sub = sub[sub['age_group'] == age_filter]
        if cat:
            sub = sub[sub[col].astype(str) != '999']
            return (sub[col] == cat).mean() * 100 if len(sub) else float('nan')
        return sub[col].mean()

    def _annotate(ax, bars, vals, is_pct):
        for bar, val in zip(bars, vals):
            if pd.isna(val) or val == 0:
                continue
            txt = f'{val:.0f}%' if is_pct else f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                    txt, ha='center', va='bottom', fontsize=7)

    for slug, label, col, cat in OutCfg:
        is_pct = cat is not None
        gender_colors = [_GENDER_LINE_COLORS.get(g, '#999') for g in _GENDER_ORDER]
        age_colors    = [_AGE_COLORS.get(a, '#999') for a in _AGE_ORDER]

        # ── GENDER plot ──────────────────────────────────────────────────────
        fig_g, axes_g = plt.subplots(2, 4, figsize=(22, 10), sharey=False)
        for ri, model in enumerate(models):
            mb  = base[base['model'] == model]
            mna = neu_age[neu_age['model'] == model]
            mfl = neu_full[neu_full['model'] == model]

            for ci, sev in enumerate(_SEVERITY_ORDER):
                ax = axes_g[ri, ci]
                sb  = mb[mb['severity_level'] == sev]
                sna = mna[mna['severity_level'] == sev]
                sfl = mfl[mfl['severity_level'] == sev]

                # neutral_full reference line (no labels at all)
                ref_line = _val(sfl, col, cat)

                base_vals = [_val(sb,  col, cat, gender_filter=g) for g in _GENDER_ORDER]
                nage_vals = [_val(sna, col, cat, gender_filter=g) for g in _GENDER_ORDER]

                x = range(len(_GENDER_ORDER))
                b_na = ax.bar([i - w / 2 for i in x], nage_vals, w, color=gender_colors,
                              alpha=0.45, hatch='//', edgecolor='white', label='neutral_age')
                b_bs = ax.bar([i + w / 2 for i in x], base_vals, w, color=gender_colors,
                              label='base')

                # auto-scale y then draw reference line
                all_vals = [v for v in base_vals + nage_vals if pd.notna(v)]
                ymin = min(all_vals + [ref_line]) * (0.92 if not is_pct else 0)
                ymax = max(all_vals + [ref_line]) * 1.15 if all_vals else 1
                if is_pct:
                    ymin, ymax = 0, min(ymax + 5, 110)
                ax.set_ylim(ymin, ymax)

                ref_label = (f'neutral_full ({ref_line:.0f}%)' if is_pct
                             else f'neutral_full ({ref_line:.2f})')
                ax.axhline(ref_line, color='black', linewidth=1.8, linestyle='--',
                           label=ref_label)

                _annotate(ax, b_na, nage_vals, is_pct)
                _annotate(ax, b_bs, base_vals, is_pct)

                ax.set_xticks(list(x))
                ax.set_xticklabels(_GENDER_ORDER, rotation=25, ha='right', fontsize=8)
                if is_pct:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
                ax.set_title(sev, fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f'{model.split(":")[0]}\n{label}', fontsize=9)
                if ri == 0 and ci == 3:
                    ax.legend(fontsize=8, loc='upper right')

        fig_g.suptitle(
            f'{label}: base (solid) vs neutral_age (hatched) per gender\n'
            'Dashed line = neutral_full (no labels at all)', fontsize=12)
        fig_g.tight_layout()
        _save_fig(fig_g, plots_dir / f'neutral_vs_base_{slug}_gender.png')

        # ── AGE plot ─────────────────────────────────────────────────────────
        fig_a, axes_a = plt.subplots(2, 4, figsize=(18, 10), sharey=False)
        for ri, model in enumerate(models):
            mb  = base[base['model'] == model]
            mng = neu_gender[neu_gender['model'] == model]
            mfl = neu_full[neu_full['model'] == model]

            for ci, sev in enumerate(_SEVERITY_ORDER):
                ax = axes_a[ri, ci]
                sb  = mb[mb['severity_level'] == sev]
                sng = mng[mng['severity_level'] == sev]
                sfl = mfl[mfl['severity_level'] == sev]

                # neutral_full reference line (no gender, no age)
                ref_line = _val(sfl, col, cat)

                base_vals = [_val(sb,  col, cat, age_filter=a) for a in _AGE_ORDER]
                ng_vals   = [_val(sng, col, cat, age_filter=a) for a in _AGE_ORDER]

                x = range(len(_AGE_ORDER))
                b_ng = ax.bar([i - w / 2 for i in x], ng_vals, w, color=age_colors,
                              alpha=0.45, hatch='//', edgecolor='white', label='neutral_gender')
                b_bs = ax.bar([i + w / 2 for i in x], base_vals, w, color=age_colors,
                              label='base')

                all_vals = [v for v in base_vals + ng_vals if pd.notna(v)]
                ymin = min(all_vals + [ref_line]) * (0.92 if not is_pct else 0)
                ymax = max(all_vals + [ref_line]) * 1.15 if all_vals else 1
                if is_pct:
                    ymin, ymax = 0, min(ymax + 5, 110)
                ax.set_ylim(ymin, ymax)

                ref_label = (f'neutral_full ({ref_line:.0f}%)' if is_pct
                             else f'neutral_full ({ref_line:.2f})')
                ax.axhline(ref_line, color='black', linewidth=1.8, linestyle='--',
                           label=ref_label)

                _annotate(ax, b_ng, ng_vals, is_pct)
                _annotate(ax, b_bs, base_vals, is_pct)

                ax.set_xticks(list(x))
                ax.set_xticklabels(_AGE_ORDER, fontsize=9)
                if is_pct:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
                ax.set_title(sev, fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f'{model.split(":")[0]}\n{label}', fontsize=9)
                if ri == 0 and ci == 3:
                    ax.legend(fontsize=8, loc='upper right')

        fig_a.suptitle(
            f'{label}: base (solid) vs neutral_gender (hatched) per age group\n'
            'Dashed line = neutral_full (no gender, no age)', fontsize=12)
        fig_a.tight_layout()
        _save_fig(fig_a, plots_dir / f'neutral_vs_base_{slug}_age.png')


def _plot_mitigation_effects_all(df: pd.DataFrame, plots_dir: Path) -> None:
    """One 2×4 grouped-bar figure per outcome for gender AND age dimensions.

    Gender plots  (file: mitigation_vs_base_<outcome>_gender.png)
      x = genders, solid = base, hatched = mitigation

    Age plots  (file: mitigation_vs_base_<outcome>_age.png)
      x = age groups, solid = base, hatched = mitigation

    Each figure: 2 rows (models) × 4 cols (severities).
    Outcomes: 4 continuous scores + Q1 diagnosis + Q4b pharmacological + Q6 medicalized.
    """
    cont = ['severity_score', 'impairment_score', 'treatment_score', 'minimizing_score']

    base = df[df['prompt_type'] == 'base'].copy()
    mit  = df[df['prompt_type'] == 'mitigation'].copy()

    for c in cont:
        for d in (base, mit):
            d[c] = pd.to_numeric(d[c], errors='coerce')
            d.loc[d[c] == 999, c] = float('nan')

    models = ['llama3:8b', 'mistral:7b']
    w = 0.38

    # diagnosis: use diagnosis_yes_no (1 = yes); treat as % diagnosed
    OutCfg = [
        ('severity_score',      'severity score',        'severity_score',         None),
        ('impairment_score',    'impairment score',      'impairment_score',        None),
        ('treatment_score',     'treatment score',       'treatment_score',         None),
        ('minimizing_score',    'minimizing score',      'minimizing_score',        None),
        ('q1_diagnosis',        'Q1 diagnosis %',        'diagnosis_yes_no',        1),
        ('q4b_pharmacological', 'Q4b pharmacological %', 'treatment_type_profile', 'pharmacological'),
        ('q6_medicalized',      'Q6 medicalized %',      'framing_label',          'medicalized'),
    ]

    def _val(d, col, cat, gender_filter=None, age_filter=None):
        sub = d.copy()
        if gender_filter is not None:
            sub = sub[sub['gender'] == gender_filter]
        if age_filter is not None:
            sub = sub[sub['age_group'] == age_filter]
        if cat is not None:
            sub = sub[sub[col].astype(str) != '999']
            if len(sub) == 0:
                return float('nan')
            if isinstance(cat, int):
                sub[col] = pd.to_numeric(sub[col], errors='coerce')
            return (sub[col] == cat).mean() * 100
        return sub[col].mean()

    def _annotate(ax, bars, vals, is_pct, yrange):
        for bar, val in zip(bars, vals):
            if pd.isna(val):
                continue
            txt = f'{val:.0f}%' if is_pct else f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + yrange * 0.02,
                    txt, ha='center', va='bottom', fontsize=7)

    def _ylim(vals_all, is_pct):
        all_v = [v for v in vals_all if pd.notna(v)]
        if not all_v:
            return (0, 1)
        lo = min(all_v) * (0.92 if not is_pct else 0)
        hi = max(all_v) * 1.15
        if is_pct:
            return (0, min(hi + 5, 110))
        return (lo, hi)

    for slug, label, col, cat in OutCfg:
        is_pct = cat is not None
        gender_colors = [_GENDER_LINE_COLORS.get(g, '#999') for g in _GENDER_ORDER]
        age_colors    = [_AGE_COLORS.get(a, '#999') for a in _AGE_ORDER]

        # ── GENDER plot ──────────────────────────────────────────────────────
        fig_g, axes_g = plt.subplots(2, 4, figsize=(22, 10), sharey=False)
        for ri, model in enumerate(models):
            mb = base[base['model'] == model]
            mm = mit[mit['model'] == model]

            for ci, sev in enumerate(_SEVERITY_ORDER):
                ax = axes_g[ri, ci]
                sb = mb[mb['severity_level'] == sev]
                sm = mm[mm['severity_level'] == sev]

                base_vals = [_val(sb, col, cat, gender_filter=g) for g in _GENDER_ORDER]
                mit_vals  = [_val(sm, col, cat, gender_filter=g) for g in _GENDER_ORDER]

                x = range(len(_GENDER_ORDER))
                ylo, yhi = _ylim(base_vals + mit_vals, is_pct)
                ax.set_ylim(ylo, yhi)

                b_mt = ax.bar([i - w / 2 for i in x], mit_vals, w, color=gender_colors,
                              alpha=0.45, hatch='//', edgecolor='white', label='mitigation')
                b_bs = ax.bar([i + w / 2 for i in x], base_vals, w, color=gender_colors,
                              label='base')

                yr = yhi - ylo
                _annotate(ax, b_mt, mit_vals, is_pct, yr)
                _annotate(ax, b_bs, base_vals, is_pct, yr)

                ax.set_xticks(list(x))
                ax.set_xticklabels(_GENDER_ORDER, rotation=25, ha='right', fontsize=8)
                if is_pct:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
                ax.set_title(sev, fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f'{model.split(":")[0]}\n{label}', fontsize=9)
                if ri == 0 and ci == 3:
                    ax.legend(fontsize=8, loc='upper right')

        fig_g.suptitle(f'{label}: base (solid) vs mitigation (hatched) per gender', fontsize=12)
        fig_g.tight_layout()
        _save_fig(fig_g, plots_dir / f'mitigation_vs_base_{slug}_gender.png')

        # ── AGE plot ─────────────────────────────────────────────────────────
        fig_a, axes_a = plt.subplots(2, 4, figsize=(18, 10), sharey=False)
        for ri, model in enumerate(models):
            mb = base[base['model'] == model]
            mm = mit[mit['model'] == model]

            for ci, sev in enumerate(_SEVERITY_ORDER):
                ax = axes_a[ri, ci]
                sb = mb[mb['severity_level'] == sev]
                sm = mm[mm['severity_level'] == sev]

                base_vals = [_val(sb, col, cat, age_filter=a) for a in _AGE_ORDER]
                mit_vals  = [_val(sm, col, cat, age_filter=a) for a in _AGE_ORDER]

                x = range(len(_AGE_ORDER))
                ylo, yhi = _ylim(base_vals + mit_vals, is_pct)
                ax.set_ylim(ylo, yhi)

                b_mt = ax.bar([i - w / 2 for i in x], mit_vals, w, color=age_colors,
                              alpha=0.45, hatch='//', edgecolor='white', label='mitigation')
                b_bs = ax.bar([i + w / 2 for i in x], base_vals, w, color=age_colors,
                              label='base')

                yr = yhi - ylo
                _annotate(ax, b_mt, mit_vals, is_pct, yr)
                _annotate(ax, b_bs, base_vals, is_pct, yr)

                ax.set_xticks(list(x))
                ax.set_xticklabels(_AGE_ORDER, fontsize=9)
                if is_pct:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
                ax.set_title(sev, fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f'{model.split(":")[0]}\n{label}', fontsize=9)
                if ri == 0 and ci == 3:
                    ax.legend(fontsize=8, loc='upper right')

        fig_a.suptitle(f'{label}: base (solid) vs mitigation (hatched) per age group', fontsize=12)
        fig_a.tight_layout()
        _save_fig(fig_a, plots_dir / f'mitigation_vs_base_{slug}_age.png')


def _plot_neutral_diagnosis_ambiguous(df: pd.DataFrame, plots_dir: Path) -> None:
    """2×2 figure: Q1 diagnosis rate, ambiguous vignette only, base vs neutral.
    Rows = models, cols = gender dimension | age dimension.
    Gender panels: solid=base, hatched=neutral_age, dashed=neutral_gender.
    Age panels:    solid=base, hatched=neutral_gender, dashed=neutral_full.
    """
    amb = df[df['severity_level'] == 'ambiguous'].copy()

    def _diag_pct(d, gender_filter=None, age_filter=None):
        sub = d.copy()
        if gender_filter is not None:
            sub = sub[sub['gender'] == gender_filter]
        if age_filter is not None:
            sub = sub[sub['age_group'] == age_filter]
        sub = sub[sub['diagnosis_yes_no'].astype(str) != '999']
        sub['diagnosis_yes_no'] = pd.to_numeric(sub['diagnosis_yes_no'], errors='coerce')
        return sub['diagnosis_yes_no'].mean() * 100 if len(sub) else float('nan')

    models = ['llama3:8b', 'mistral:7b']
    w = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ri, model in enumerate(models):
        sub = amb[amb['model'] == model]
        base_s = sub[sub['prompt_type'] == 'base']
        nage_s = sub[sub['prompt_type'] == 'neutral_age']
        ngen_s = sub[sub['prompt_type'] == 'neutral_gender']
        nful_s = sub[sub['prompt_type'] == 'neutral_full']

        # ── Gender panel ──────────────────────────────────────────────────────
        ax = axes[ri, 0]
        base_vals = [_diag_pct(base_s, gender_filter=g) for g in _GENDER_ORDER]
        nage_vals = [_diag_pct(nage_s, gender_filter=g) for g in _GENDER_ORDER]
        ref_nf    = _diag_pct(nful_s)          # neutral_full scalar (no labels)
        colors    = [_GENDER_LINE_COLORS.get(g, '#999') for g in _GENDER_ORDER]

        x = range(len(_GENDER_ORDER))
        all_vals = [v for v in base_vals + nage_vals + [ref_nf] if pd.notna(v)]
        yhi = min(max(all_vals) * 1.15 + 3, 110) if all_vals else 110
        ax.set_ylim(0, yhi)

        b_na = ax.bar([i - w / 2 for i in x], nage_vals, w, color=colors,
                      alpha=0.45, hatch='//', edgecolor='white', label='neutral_age')
        b_bs = ax.bar([i + w / 2 for i in x], base_vals, w, color=colors, label='base')
        ax.axhline(ref_nf, color='black', linewidth=1.8, linestyle='--',
                   label=f'neutral_full ({ref_nf:.0f}%)')

        yr = yhi
        for bars, vals in [(b_na, nage_vals), (b_bs, base_vals)]:
            for bar, val in zip(bars, vals):
                if pd.notna(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + yr * 0.02,
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(_GENDER_ORDER, rotation=25, ha='right', fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.set_ylabel(f'{model.split(":")[0]}\n% diagnosed', fontsize=10)
        ax.set_title('by gender' if ri == 0 else '', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')

        # ── Age panel ─────────────────────────────────────────────────────────
        ax = axes[ri, 1]
        base_vals = [_diag_pct(base_s, age_filter=a) for a in _AGE_ORDER]
        ngen_vals = [_diag_pct(ngen_s, age_filter=a) for a in _AGE_ORDER]
        ref_nf    = _diag_pct(nful_s)          # neutral_full scalar
        colors    = [_AGE_COLORS.get(a, '#999') for a in _AGE_ORDER]

        x = range(len(_AGE_ORDER))
        all_vals = [v for v in base_vals + ngen_vals + [ref_nf] if pd.notna(v)]
        yhi = min(max(all_vals) * 1.15 + 3, 110) if all_vals else 110
        ax.set_ylim(0, yhi)

        b_ng = ax.bar([i - w / 2 for i in x], ngen_vals, w, color=colors,
                      alpha=0.45, hatch='//', edgecolor='white', label='neutral_gender')
        b_bs = ax.bar([i + w / 2 for i in x], base_vals, w, color=colors, label='base')
        ax.axhline(ref_nf, color='black', linewidth=1.8, linestyle='--',
                   label=f'neutral_full ({ref_nf:.0f}%)')

        yr = yhi
        for bars, vals in [(b_ng, ngen_vals), (b_bs, base_vals)]:
            for bar, val in zip(bars, vals):
                if pd.notna(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + yr * 0.02,
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(_AGE_ORDER, fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.set_title('by age group' if ri == 0 else '', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle(
        'Q1 diagnosis rate — ambiguous vignette: base (solid) vs neutral (hatched)\n'
        'Dashed line = neutral without the demographic dimension shown', fontsize=12)
    fig.tight_layout()
    _save_fig(fig, plots_dir / 'neutral_vs_base_q1_diagnosis_ambiguous.png')


def _plot_neutral_overview(df: pd.DataFrame, plots_dir: Path) -> None:
    """Deviation heatmap (base − neutral_full) across all key outcomes × genders.
    neutral_full has no gender dimension, so each cell shows how much adding full
    demographic labels shifts the score relative to the no-label baseline.
    Mirrors _plot_mitigation_overview for direct side-by-side comparison."""
    import numpy as np

    base_g = df[(df['prompt_type'] == 'base') &
                df['gender'].isin(_GENDER_ORDER) &
                df['age_group'].isin(_AGE_ORDER)].copy()
    nf_g   = df[(df['prompt_type'] == 'neutral_full')].copy()

    cont = ['severity_score', 'impairment_score', 'treatment_score', 'minimizing_score']
    for out in cont:
        for d in (base_g, nf_g):
            d[out] = pd.to_numeric(d[out], errors='coerce')
            d.loc[d[out] == 999, out] = float('nan')

    out_labels = ['severity\nscore', 'impairment\nscore', 'treatment\nscore', 'minimizing\nscore',
                  'Q4b\npharmacological', 'Q6\nmedicalized']
    models = ['llama3:8b', 'mistral:7b']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, model in zip(axes, models):
        mb = base_g[base_g['model'] == model]
        nf = nf_g[nf_g['model'] == model]

        mat = []
        for out in cont:
            nf_mean = nf[out].mean()
            row = [mb[mb['gender'] == g][out].mean() - nf_mean for g in _GENDER_ORDER]
            mat.append(row)

        # Q4b pharmacological: base % per gender minus neutral_full %
        nf_pharm_df = nf[nf['treatment_type_profile'].astype(str) != '999']
        nf_pharm = (nf_pharm_df['treatment_type_profile'] == 'pharmacological').mean() * 100 if len(nf_pharm_df) else 0
        row_pharm = []
        for g in _GENDER_ORDER:
            gdf = mb[(mb['gender'] == g) & (mb['treatment_type_profile'].astype(str) != '999')]
            b_pct = (gdf['treatment_type_profile'] == 'pharmacological').mean() * 100 if len(gdf) else 0
            row_pharm.append(b_pct - nf_pharm)
        mat.append(row_pharm)

        # Q6 medicalized: base % per gender minus neutral_full %
        nf_med_df = nf[nf['framing_label'].astype(str) != '999']
        nf_med = (nf_med_df['framing_label'] == 'medicalized').mean() * 100 if len(nf_med_df) else 0
        row_med = []
        for g in _GENDER_ORDER:
            gdf = mb[(mb['gender'] == g) & (mb['framing_label'].astype(str) != '999')]
            b_pct = (gdf['framing_label'] == 'medicalized').mean() * 100 if len(gdf) else 0
            row_med.append(b_pct - nf_med)
        mat.append(row_med)

        mat = np.array(mat)
        vmax = max(abs(mat).max(), 5)
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(_GENDER_ORDER)))
        ax.set_yticks(range(len(out_labels)))
        ax.set_xticklabels(_GENDER_ORDER, rotation=20, ha='right')
        ax.set_yticklabels(out_labels)
        ax.set_title(model, fontsize=11)
        for i in range(len(out_labels)):
            for j in range(len(_GENDER_ORDER)):
                val = mat[i, j]
                fmt = f'{val:+.0f}pp' if i >= 4 else f'{val:+.2f}'
                ax.text(j, i, fmt, ha='center', va='center',
                        color='white' if abs(val) > vmax * 0.55 else 'black', fontsize=9)
        fig.colorbar(im, ax=ax, label='base − neutral_full')
    fig.suptitle('Base vs neutral_full: demographic label effect across all outcomes by gender',
                 fontsize=12)
    fig.tight_layout()
    _save_fig(fig, plots_dir / 'neutral_vs_base_deviation_heatmap.png')


def _plot_mitigation_overview(df: pd.DataFrame, plots_dir: Path) -> None:
    """Full deviation heatmap (base − mitigation) across all key outcomes × genders,
    mirroring _plot_neutral_overview for direct comparison."""
    import numpy as np

    base_g = df[(df['prompt_type'] == 'base') &
                df['gender'].isin(_GENDER_ORDER) &
                df['age_group'].isin(_AGE_ORDER)].copy()
    mit_g  = df[(df['prompt_type'] == 'mitigation') &
                df['gender'].isin(_GENDER_ORDER) &
                df['age_group'].isin(_AGE_ORDER)].copy()

    cont = ['severity_score', 'impairment_score', 'treatment_score', 'minimizing_score']
    for out in cont:
        for d in (base_g, mit_g):
            d[out] = pd.to_numeric(d[out], errors='coerce')
            d.loc[d[out] == 999, out] = float('nan')

    out_labels = ['severity\nscore', 'impairment\nscore', 'treatment\nscore', 'minimizing\nscore',
                  'Q4b\npharmacological', 'Q6\nmedicalized']
    models = ['llama3:8b', 'mistral:7b']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, model in zip(axes, models):
        mb = base_g[base_g['model'] == model]
        mm = mit_g[mit_g['model'] == model]

        mat = []
        for out in cont:
            row = [mm[mm['gender'] == g][out].mean() - mb[mb['gender'] == g][out].mean()
                   for g in _GENDER_ORDER]
            mat.append(row)
        # Q4b pharmacological
        row_pharm = []
        for g in _GENDER_ORDER:
            def _pct(d, g_):
                gdf = d[(d['gender'] == g_) & (d['treatment_type_profile'].astype(str) != '999')]
                return (gdf['treatment_type_profile'] == 'pharmacological').mean() * 100 if len(gdf) else 0
            row_pharm.append(_pct(mm, g) - _pct(mb, g))
        mat.append(row_pharm)
        # Q6 medicalized
        row_med = []
        for g in _GENDER_ORDER:
            def _mpct(d, g_):
                gdf = d[(d['gender'] == g_) & (d['framing_label'].astype(str) != '999')]
                return (gdf['framing_label'] == 'medicalized').mean() * 100 if len(gdf) else 0
            row_med.append(_mpct(mm, g) - _mpct(mb, g))
        mat.append(row_med)

        mat = np.array(mat)
        vmax = max(abs(mat).max(), 5)
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(_GENDER_ORDER)))
        ax.set_yticks(range(len(out_labels)))
        ax.set_xticklabels(_GENDER_ORDER, rotation=20, ha='right')
        ax.set_yticklabels(out_labels)
        ax.set_title(model, fontsize=11)
        for i in range(len(out_labels)):
            for j in range(len(_GENDER_ORDER)):
                val = mat[i, j]
                fmt = f'{val:+.0f}pp' if i >= 4 else f'{val:+.2f}'
                ax.text(j, i, fmt, ha='center', va='center',
                        color='white' if abs(val) > vmax * 0.55 else 'black', fontsize=9)
        fig.colorbar(im, ax=ax, label='mitigation − base')
    fig.suptitle('Base vs mitigation: deviation across all outcomes by gender', fontsize=12)
    fig.tight_layout()
    _save_fig(fig, plots_dir / 'mitigation_deviation_heatmap.png')


def run() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()
    if df.empty:
        print("=== analysis.py DONE ===  (no data)", flush=True)
        return

    print(f"=== analysis.py START ===  {len(df)} rows loaded from data/processed/results.csv",
          flush=True)

    _save_csv(_aggregated(df), _RESULTS_DIR / 'aggregated_results.csv')
    _save_csv(_comparisons_gender(df), _RESULTS_DIR / 'comparisons_gender.csv')
    _save_csv(_comparisons_age(df), _RESULTS_DIR / 'comparisons_age.csv')
    _save_csv(_comparisons_intersectionality(df), _RESULTS_DIR / 'comparisons_intersectionality.csv')

    _plot_treatment_type_q4b(df, _PLOTS_DIR)
    _plot_framing(df, _PLOTS_DIR)
    _plot_framing_medicalization(df, _PLOTS_DIR)
    _plot_diagnosis_ambiguous(df, _PLOTS_DIR)
    _plot_diagnosis_mitigation(df, _PLOTS_DIR)
    _plot_score_effects(df, _PLOTS_DIR, 'severity_score')
    _plot_score_effects(df, _PLOTS_DIR, 'impairment_score')
    _plot_score_effects(df, _PLOTS_DIR, 'treatment_score')
    _plot_score_effects(df, _PLOTS_DIR, 'minimizing_score')
    _plot_neutral_effects(df, _PLOTS_DIR)
    _plot_neutral_diagnosis_ambiguous(df, _PLOTS_DIR)
    _plot_neutral_overview(df, _PLOTS_DIR)
    _plot_mitigation_effects(df, _PLOTS_DIR)
    _plot_mitigation_effects_all(df, _PLOTS_DIR)
    _plot_mitigation_overview(df, _PLOTS_DIR)

    print("=== analysis.py DONE ===", flush=True)
