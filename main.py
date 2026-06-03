#!/usr/bin/env python3
"""
main.py — single entry point for the mental health bias detection experiment.

Usage (design.md § main.py CLI):
  python main.py                                # full experiment (~3120 calls)
  python main.py --prototype                    # mild severity, both models (~780 calls)
  python main.py --prototype --models llama3:8b # mild severity, llama3 only (~390 calls)
  python main.py --prototype --severity moderate # override severity level
  python main.py --models llama3:8b            # llama3 only, all severities
  python main.py --severity mild moderate       # subset of severities

  # Step-by-step (resume from checkpoint — each step reads from disk):
  python main.py --step run                    # runner only → data/raw_responses/
  python main.py --step parse                  # parser + scorer → data/processed/results.csv
  python main.py --step analyse                # analysis → results/
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root is the directory containing this file
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.runner   import run as _runner_run
from src.runner   import ALL_MODELS, ALL_SEVERITIES, ALL_PROMPT_TYPES, ALL_GENDERS, ALL_AGE_GROUPS
from src.parser   import parse_all
from src.scorer   import score
from src.analysis import run as _analysis_run

_RAW_DIR       = _ROOT / "data" / "raw_responses"
_PROCESSED_DIR = _ROOT / "data" / "processed"
_RESULTS_DIR   = _ROOT / "results"

# Exact column order from design.md § Output Schema
_RESULTS_COLUMNS = [
    "variant_id", "prompt_type", "gender", "age_group", "severity_level",
    "run_number", "model", "temperature", "timestamp", "raw_response",
    "diagnosis_label", "diagnosis_yes_no",
    "severity_score", "impairment_score",
    "treatment_score", "treatment_type_profile",
    "minimizing_score",
    "diagnostic_evidence", "framing_label",
    "gendered_count", "medicalized_count", "distanced_count", "neutral_count",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Gender & Age Bias in LLM-Based Depression Diagnosis — experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --prototype --models llama3:8b   # prototype verification\n"
            "  python main.py --step parse                      # re-parse existing raw files\n"
            "  python main.py                                   # full experiment (3120 calls)\n"
        ),
    )
    p.add_argument(
        "--prototype", action="store_true",
        help="Prototype scope: one severity level (default mild). "
             "Use --severity to override the level.",
    )
    p.add_argument(
        "--severity", nargs="+",
        choices=["mild", "moderate", "severe", "ambiguous"],
        metavar="LEVEL",
        help="Severity level(s) to run. Overrides --prototype default.",
    )
    p.add_argument(
        "--models", nargs="+",
        choices=["llama3:8b", "mistral:7b"],
        metavar="MODEL",
        help="Ollama model(s) to use (default: both).",
    )
    p.add_argument(
        "--step", choices=["run", "parse", "analyse"],
        help=(
            "Run only one pipeline step:\n"
            "  run     — call Ollama → data/raw_responses/\n"
            "  parse   — parse + score raw_responses/ → data/processed/results.csv\n"
            "  analyse — results.csv → results/\n"
            "Omit to run all three steps in sequence."
        ),
    )
    p.add_argument(
        "--runs", type=int, default=10, metavar="N",
        help="Runs per variant (default: 10).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Completeness check (requirements.md § Acceptance Criteria)
# ─────────────────────────────────────────────────────────────────────────────

_SCORED_FIELDS = [
    "diagnosis_yes_no",      # Q1
    "severity_score",        # Q2
    "impairment_score",      # Q3
    "treatment_score",       # Q4a
    "treatment_type_profile",# Q4b
    "minimizing_score",      # Q5
]


def _check_completeness(df: pd.DataFrame) -> None:
    """
    Report 999-rate per scored field (requirements.md § Acceptance Criteria).
    Threshold: < 5 %. Exceeding it means the prompt or parser needs fixing.
    """
    print("\nCompleteness check (requirement: < 5 % 999s per field):")
    any_fail = False
    for col in _SCORED_FIELDS:
        if col not in df.columns:
            print(f"  SKIP  {col}: column absent")
            continue
        rate = (df[col] == 999).mean() * 100
        status = "FAIL" if rate > 5 else "OK  "
        print(f"  {status}  {col}: {rate:.1f}% 999s  (n={len(df)})")
        if rate > 5:
            any_fail = True
    if any_fail:
        print(
            "\n  WARNING: one or more fields exceed the 5% threshold.\n"
            "  Fix the prompt format or parser before running the full experiment."
        )
    else:
        print("  All fields within threshold.")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def _step_run(severities: list[str], models: list[str], runs: int) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP 1/3 — Runner")
    print(f"  severities={severities}  models={models}  runs={runs}")
    print(f"{'='*60}")
    _runner_run(
        severities       = severities,
        prompt_types     = ALL_PROMPT_TYPES,
        genders          = ALL_GENDERS,
        age_groups       = ALL_AGE_GROUPS,
        models           = models,
        runs_per_variant = runs,
        output_dir       = _RAW_DIR,
    )


def _step_parse(models: list[str]) -> None:
    """
    Parse all raw_responses/ on disk and score with LLM-as-judge.
    Scope flags (severity, models) are intentionally ignored here — the step
    processes whatever is already on disk (design.md: 'resume from checkpoint').
    The `models` list is passed to scorer.score() only to determine the
    cross-model judge mapping.
    """
    print(f"\n{'='*60}")
    print(f"  STEP 2/3 — Parser + Scorer")
    print(f"{'='*60}")
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing raw responses…")
    df = parse_all(raw_dir=_RAW_DIR)
    print(f"  Parsed {len(df)} rows from {_RAW_DIR.name}/")

    print("Scoring treatment recommendations (LLM-as-judge)…")
    # Use unique models found in the data for cross-model judge mapping
    available = list(df["model"].unique()) if len(df) else models
    df = score(df, available_models=available)

    # Enforce column order from design.md § Output Schema; drops treatment_text
    df = df.reindex(columns=[c for c in _RESULTS_COLUMNS if c in df.columns])

    out_path = _PROCESSED_DIR / "results.csv"
    df.to_csv(out_path, index=False)
    print(f"  Written {out_path}  ({len(df)} rows × {len(df.columns)} columns)")

    _check_completeness(df)


def _step_analyse() -> None:
    print(f"\n{'='*60}")
    print(f"  STEP 3/3 — Analysis")
    print(f"{'='*60}")
    _analysis_run(
        results_path = _PROCESSED_DIR / "results.csv",
        output_dir   = _RESULTS_DIR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_args()

    # ── Resolve experiment scope ──────────────────────────────────────────────
    if args.severity:
        severities = args.severity
    elif args.prototype:
        severities = ["mild"]
    else:
        severities = ALL_SEVERITIES

    models = args.models if args.models else ALL_MODELS
    step   = args.step

    scope_tag = "prototype" if (args.prototype and not args.severity) else "custom"
    print(
        f"Mental Health Bias Detection — {scope_tag} scope\n"
        f"  severities : {severities}\n"
        f"  models     : {models}\n"
        f"  step       : {step or 'all (run → parse → analyse)'}\n"
        f"  runs/variant: {args.runs}"
    )

    # ── Execute pipeline ──────────────────────────────────────────────────────
    if step is None or step == "run":
        _step_run(severities, models, args.runs)

    if step is None or step == "parse":
        _step_parse(models)

    if step is None or step == "analyse":
        _step_analyse()

    print("\nDone.")


if __name__ == "__main__":
    main()
