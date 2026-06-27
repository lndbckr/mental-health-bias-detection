"""
main.py — single entry point for the mental-health-bias-detection pipeline.
Implements: design.md § main.py CLI, § Batched Runs, § Progress Output.
"""

import argparse
import sys
import time
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

_ALL_SEVERITIES    = ['mild', 'moderate', 'severe', 'ambiguous']
_ALL_MODELS        = ['llama3:8b', 'mistral:7b']
_ALL_PROMPT_TYPES  = ['base', 'mitigation', 'neutral_full', 'neutral_age', 'neutral_gender']

_RESULTS_COLUMNS = [
    'variant_id', 'prompt_type', 'gender', 'age_group', 'severity_level',
    'run_number', 'model', 'temperature', 'timestamp', 'raw_response',
    'diagnosis_label', 'diagnosis_yes_no',
    'severity_score', 'impairment_score',
    'treatment_score', 'treatment_type_profile',
    'minimizing_score',
    'diagnostic_evidence', 'framing_label',
    'gendered_count', 'medicalized_count', 'distanced_count', 'neutral_count',
]

_PROCESSED_DIR = Path('data') / 'processed'


def _model_slug(tag: str) -> str:
    return tag.split(':')[0]


def _default_judge(models: list[str]) -> str:
    """Cross-model judge: if one model, use the other; if both, each judges the other (return one for batch)."""
    other = [m for m in _ALL_MODELS if m not in models]
    if other:
        return other[0]
    # Both models running — caller handles per-model judgement
    return models[0]


# ── step: run ────────────────────────────────────────────────────────────────

def _step_run(severities: list[str], models: list[str],
              prompt_types: list[str], runs: int) -> None:
    from src import runner
    runner.run(severities=severities, models=models,
               prompt_types=prompt_types, runs_per_variant=runs)


# ── step: parse (+score) ─────────────────────────────────────────────────────

def _step_parse(severities: list[str], models: list[str],
                judge_model: str, raw_dir: Path | None = None) -> None:
    import pandas as pd
    from src import parser, scorer

    df = parser.run(raw_dir=raw_dir)

    if df.empty:
        print("  WARN: parser produced empty DataFrame", file=sys.stderr, flush=True)
        return

    # Filter to only the requested severities and models if specified
    if 'severity_level' in df.columns:
        df = df[df['severity_level'].isin(severities)].copy()
    if 'model' in df.columns:
        df = df[df['model'].isin(models)].copy()

    # Initialize treatment columns with correct dtypes per design.md § Column Notes
    df['treatment_score'] = 999.0
    df['treatment_type_profile'] = pd.array([999] * len(df), dtype=object)

    df = scorer.run(df, judge_model=judge_model)

    # Drop Q1=999 rows before writing
    df = df[df['diagnosis_yes_no'] != 999].copy()

    # Write one partition file per severity × model slug
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for sev in df['severity_level'].unique():
        for model in df['model'].unique():
            part = df[(df['severity_level'] == sev) & (df['model'] == model)].copy()
            if part.empty:
                continue
            slug = _model_slug(model)
            out_path = _PROCESSED_DIR / f"results_{sev}_{slug}.csv"
            # drop treatment_text before writing (intermediate column)
            out_cols = [c for c in _RESULTS_COLUMNS if c in part.columns]
            part = part.reindex(columns=out_cols)
            part.to_csv(out_path, index=False)
            print(f"  → wrote {out_path}  ({len(part)} rows)", flush=True)


# ── step: analyse ─────────────────────────────────────────────────────────────

def _step_analyse() -> None:
    from src import analysis
    analysis.run()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Mental-health bias detection pipeline.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--severity', nargs='+', choices=_ALL_SEVERITIES,
                   default=None,
                   help='Severity levels to process (default: all four)')
    p.add_argument('--models', nargs='+', choices=_ALL_MODELS,
                   default=None,
                   help='Experiment models (default: both)')
    p.add_argument('--judge-model', dest='judge_model', default=None,
                   help='LLM judge for Q4a/Q4b (default: cross-model)')
    p.add_argument('--step', choices=['run', 'parse', 'analyse'],
                   default=None,
                   help='Run only one pipeline step')
    p.add_argument('--batch', action='store_true',
                   help='Run + parse for one nightly partition; skip analysis')
    p.add_argument('--prototype', action='store_true',
                   help='Single severity (mild unless --severity overrides), full matrix, 1 run')
    p.add_argument('--runs', type=int, default=10,
                   help='Runs per variant (default: 10)')
    return p


def main() -> None:
    _batch_start = time.time()

    args = _build_parser().parse_args()

    # ── resolve severities ────────────────────────────────────────────────────
    if args.severity:
        severities = args.severity
    elif args.prototype:
        severities = ['mild']
    else:
        severities = _ALL_SEVERITIES

    # ── resolve models ────────────────────────────────────────────────────────
    models = args.models if args.models else _ALL_MODELS

    # ── resolve judge model ───────────────────────────────────────────────────
    if args.judge_model:
        judge_model = args.judge_model
    else:
        judge_model = _default_judge(models)

    # ── resolve prompt_types ──────────────────────────────────────────────────
    prompt_types = _ALL_PROMPT_TYPES

    # ── resolve runs per variant ──────────────────────────────────────────────
    runs = 1 if args.prototype else args.runs

    # ── execute steps ─────────────────────────────────────────────────────────
    if args.step == 'run':
        _step_run(severities, models, prompt_types, runs)

    elif args.step == 'parse':
        _step_parse(severities, models, judge_model)

    elif args.step == 'analyse':
        _step_analyse()

    elif args.batch:
        # run + parse, no analyse
        _step_run(severities, models, prompt_types, runs)
        _step_parse(severities, models, judge_model)

    else:
        # full pipeline
        _step_run(severities, models, prompt_types, runs)
        _step_parse(severities, models, judge_model)
        _step_analyse()

    elapsed = int(time.time() - _batch_start)
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)
    print(f"=== BATCH COMPLETE ===  total time: {h}h {m}m {s}s", flush=True)


if __name__ == '__main__':
    main()
