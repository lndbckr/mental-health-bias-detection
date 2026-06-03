"""
src/runner.py
Implements design.md § Architecture → runner.py

Calls Ollama HTTP API for each (variant × model × run) and writes a raw
response JSON to data/raw_responses/. Existing files are skipped, so the
pipeline can resume after a crash without re-running completed calls.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

from src.prompt_builder import build_prompt, get_variants, make_variant_id

# ---------------------------------------------------------------------------
# Constants (design.md § Implementation Details + Experiment Variables)
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0.8

ALL_MODELS       = ["llama3:8b", "mistral:7b"]
ALL_GENDERS      = ["woman", "man", "non-binary", "trans-woman", "trans-man"]
ALL_AGE_GROUPS   = ["18-25", "40-55", "65+"]
ALL_SEVERITIES   = ["mild", "moderate", "severe", "ambiguous"]
ALL_PROMPT_TYPES = ["base", "mitigation", "neutral_full", "neutral_age", "neutral_gender"]

_RAW_DIR = Path(__file__).parent.parent / "data" / "raw_responses"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filename(variant_id: str) -> str:
    """Colons replaced with underscores for filesystem safety (design.md § Raw Response Format)."""
    return variant_id.replace(":", "_") + ".json"


def _call_ollama(model: str, prompt: str) -> str:
    """
    POST to Ollama generate endpoint (stream=False, temperature=0.8, no seed).
    Returns the raw text response string.
    Raises requests.exceptions.ConnectionError if Ollama is not running.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["response"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    severities: list[str] = ALL_SEVERITIES,
    prompt_types: list[str] = ALL_PROMPT_TYPES,
    genders: list[str] = ALL_GENDERS,
    age_groups: list[str] = ALL_AGE_GROUPS,
    models: list[str] = ALL_MODELS,
    runs_per_variant: int = 10,
    output_dir: Path | None = None,
) -> None:
    """
    Run the prompt → Ollama → raw_response pipeline for the given scope.

    Parameters map directly to the experiment dimensions in design.md.
    Prototype scope: severities=['mild'], models=['llama3:8b'].

    Each completed run is written to output_dir as:
        {variant_id with ':' → '_'}.json

    Skips files that already exist (crash-resume).
    Aborts on ConnectionError — Ollama must be running before calling run().
    """
    out_dir = output_dir or _RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = get_variants(severities, prompt_types, genders, age_groups)
    total = len(variants) * len(models) * runs_per_variant
    done = skipped = failed = 0

    print(
        f"Runner: {len(variants)} variants × {len(models)} model(s) "
        f"× {runs_per_variant} runs = {total} calls"
    )

    for model in models:
        for variant in variants:
            severity    = variant["severity"]
            prompt_type = variant["prompt_type"]
            gender_id   = variant["gender_id"]
            age_group_id= variant["age_group_id"]
            gender      = variant["gender"]
            age_group   = variant["age_group"]

            # Build prompt once per variant — reused across all N runs
            prompt = build_prompt(severity, prompt_type, gender_id, age_group_id)

            for run_number in range(1, runs_per_variant + 1):
                variant_id = make_variant_id(
                    prompt_type, gender_id, age_group_id, severity, model, run_number
                )
                out_path = out_dir / _filename(variant_id)

                if out_path.exists():
                    skipped += 1
                    done += 1
                    continue

                try:
                    raw_response = _call_ollama(model, prompt)

                except requests.exceptions.ConnectionError:
                    # Fatal: Ollama is not reachable — no point attempting remaining calls
                    print(
                        f"\nFATAL: Cannot connect to Ollama at {OLLAMA_URL}\n"
                        f"Start Ollama with 'ollama serve' and re-run.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                except requests.exceptions.Timeout:
                    print(f"  TIMEOUT  {variant_id}", file=sys.stderr)
                    failed += 1
                    done += 1
                    continue

                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR    {variant_id}: {exc}", file=sys.stderr)
                    failed += 1
                    done += 1
                    continue

                # ── Write raw response record (design.md § Raw Response Format) ──
                record = {
                    "variant_id":     variant_id,
                    "prompt_type":    prompt_type,
                    "gender":         gender,
                    "age_group":      age_group,
                    "severity_level": severity,
                    "run_number":     run_number,
                    "model":          model,
                    "temperature":    TEMPERATURE,
                    "timestamp":      datetime.now().isoformat(timespec="seconds"),
                    "prompt":         prompt,
                    "raw_response":   raw_response,
                }
                out_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                done += 1
                pct = 100 * done / total
                print(f"  [{done:>4}/{total} {pct:>3.0f}%]  {variant_id}", flush=True)

    new_calls = done - skipped - failed
    print(
        f"Runner complete — new: {new_calls}  skipped: {skipped}  failed: {failed}"
    )
