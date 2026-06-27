"""
src/runner.py
Implements: design.md § runner.py, requirements.md § CLI Progress Output.
Calls Ollama HTTP API for each variant × run, writes raw JSON to data/raw_responses/.
Crash-resume: skips any variant_id whose .json already exists.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from src.prompt_builder import (
    build_prompt,
    iter_variants,
    make_variant_id,
    model_slug,
)

_RAW_DIR  = Path(__file__).parent.parent / "data" / "raw_responses"
_OLLAMA_URL = "http://localhost:11434/api/generate"
_TEMPERATURE = 0.8
_RUNS_PER_VARIANT = 10

_ALL_PROMPT_TYPES = ['base', 'mitigation', 'neutral_full', 'neutral_age', 'neutral_gender']


def _ollama_call(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": _TEMPERATURE},
    }
    resp = requests.post(_OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


def run(severities: list[str], models: list[str],
        prompt_types: list[str] | None = None,
        runs_per_variant: int = _RUNS_PER_VARIANT) -> None:

    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    if prompt_types is None:
        prompt_types = _ALL_PROMPT_TYPES

    variants = iter_variants(severities, prompt_types)
    total_calls = len(variants) * len(models) * runs_per_variant

    print(
        f"=== runner.py START ===  severity={'+'.join(severities)}  "
        f"model={'+'.join(models)}  variants={len(variants) * len(models)}  "
        f"runs/variant={runs_per_variant}  total={total_calls} calls",
        flush=True,
    )

    call_count = 0
    skipped    = 0

    for model in models:
        slug = model_slug(model)
        for v in variants:
            gid = v['gender'] if v['gender'] != 'unspecified' else None
            aid = v['age_group'] if v['age_group'] != 'unspecified' else None

            for run_num in range(1, runs_per_variant + 1):
                vid = make_variant_id(
                    v['prompt_type'], v['gender'], v['age_group'],
                    v['severity'], slug, run_num,
                )
                out_path = _RAW_DIR / f"{vid}.json"

                if out_path.exists():
                    skipped += 1
                    call_count += 1
                    _maybe_print_progress(call_count, total_calls, vid)
                    continue

                prompt = build_prompt(
                    v['prompt_type'], v['severity'],
                    gender_id=gid, age_group_id=aid,
                )

                try:
                    raw = _ollama_call(model, prompt)
                except Exception as exc:
                    print(f"  ERROR: {vid} — {exc}", file=sys.stderr, flush=True)
                    raw = ""

                record = {
                    "variant_id":    vid,
                    "prompt_type":   v['prompt_type'],
                    "gender":        v['gender'],
                    "age_group":     v['age_group'],
                    "severity_level": v['severity'],
                    "run_number":    run_num,
                    "model":         model,
                    "temperature":   _TEMPERATURE,
                    "timestamp":     datetime.now().isoformat(timespec='seconds'),
                    "prompt":        prompt,
                    "raw_response":  raw,
                }

                out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                    encoding='utf-8')
                call_count += 1
                _maybe_print_progress(call_count, total_calls, vid)

    new_calls = call_count - skipped
    print(
        f"=== runner.py DONE  ===  {new_calls} new calls made  "
        f"({skipped}/{total_calls} skipped — raw file already on disk)",
        flush=True,
    )


def _maybe_print_progress(count: int, total: int, vid: str) -> None:
    width = len(str(total))
    if count % 50 == 0 or count == total:
        label = "DONE" if count == total else vid
        print(f"[{count:{width}d}/{total} calls]  {label}", flush=True)
