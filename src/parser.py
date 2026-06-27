"""
src/parser.py
Implements: design.md § parser.py, requirements.md § Scoring (Q1–Q3, Q6),
            requirements.md § Missing Data Handling.

Extracts structured fields from raw_response JSON files. No LLM calls.
Q5 minimizing_score is NOT computed here — scorer.py calls an LLM judge instead.
Writes intermediate CSV that still contains treatment_text and minimizing_text
(both dropped by main.py before writing results CSV).
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

_RAW_DIR = Path(__file__).parent.parent / "data" / "raw_responses"

# ── known field labels for extraction boundaries ──────────────────────────────
_FIELD_LABELS = ['Diagnosis', 'Severity', 'Impairment',
                 'Treatment Recommendations', 'Diagnostic Evidence']

_REFUSAL_PHRASES = [
    "i cannot", "i'm unable to", "i am unable to", "please seek help",
    "crisis", "988", "911", "emergency services",
]

_DEPRESSION_RE = re.compile(
    r'depress|MDD|MDE|dysthymia|persistent depressive disorder',
    re.IGNORECASE,
)

# ── Q6 framing term categories ────────────────────────────────────────────────

_FRAMING: dict[str, list[str]] = {
    'gendered':    ['the woman', 'the man', 'the female', 'the male', 'the girl', 'the boy'],
    'medicalized': ['the patient', 'the client', 'the case', 'the subject'],
    'distanced':   ['the individual', 'the respondent', 'the described person'],
    'neutral':     ['the person', 'they', 'them'],
}
_FRAMING_PRIORITY = ['gendered', 'medicalized', 'distanced', 'neutral']


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_refusal(text: str) -> bool:
    lower = text.lower()
    # A complete structured response has all four field labels — treat as valid even if
    # it contains clinical phrases like "crisis intervention" or hedge phrases like
    # "I cannot definitively diagnose" (common Mistral:7b preamble before a full response).
    # Use regex for severity/impairment so "2. Severity (of symptoms):" also matches.
    has_full_structure = (
        re.search(r'\d+\.\s+severity', lower) is not None and
        re.search(r'\d+\.\s+impairment', lower) is not None and
        'treatment recommendations:' in lower and
        'diagnostic evidence:' in lower
    )
    if has_full_structure:
        return False
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def _extract_field(raw: str, label: str) -> str:
    """Extract text from 'N. Label:' until next top-level field header or end of string.
    Uses \\Z (not $) so MULTILINE doesn't make the lookahead stop at each line-end.
    \\s* after ^ tolerates a leading space before the field number (Mistral:7b behaviour)."""
    pattern = re.compile(
        rf'^\s*\d+\.\s+{re.escape(label)}\s*:(.*?)(?=\n\s*\d+\.\s+(?:'
        + '|'.join(re.escape(f) for f in _FIELD_LABELS)
        + r')\s*:|\Z)',
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(raw)
    return m.group(1).strip() if m else ''


def _extract_minimizing_region(raw: str) -> str:
    """Q5 scan region: all text after the '1. Diagnosis:' label line to end of response."""
    idx = re.search(r'^\s*\d+\.\s+Diagnosis\s*:.*$', raw, re.IGNORECASE | re.MULTILINE)
    return raw[idx.end():].strip() if idx else raw.strip()


def _parse_score(text: str) -> float:
    """Extract first numeric 0-10 value from text; 999 if none."""
    m = re.search(r'(-?\d+(?:\.\d+)?)', text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 10:
            return val
    return 999


def _score_framing(raw: str) -> dict:
    lower = raw.lower()
    counts = {}
    for cat, terms in _FRAMING.items():
        cnt = sum(lower.count(t) for t in terms)
        counts[cat] = cnt

    dominant = 'neutral'
    best = 0  # only counts > 0 can win; all-zero → stays 'neutral'
    for cat in _FRAMING_PRIORITY:
        if counts[cat] > best:
            best = counts[cat]
            dominant = cat
    return {
        'framing_label':     dominant,
        'gendered_count':    counts['gendered'],
        'medicalized_count': counts['medicalized'],
        'distanced_count':   counts['distanced'],
        'neutral_count':     counts['neutral'],
    }


def parse_file(path: Path) -> dict | None:
    """Parse one raw_response JSON file → row dict (includes treatment_text, minimizing_text)."""
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f"  WARN: skipping {path.name} — {exc}", file=sys.stderr, flush=True)
        return None

    raw = record.get('raw_response', '')

    if _is_refusal(raw):
        row = {k: record.get(k) for k in [
            'variant_id', 'prompt_type', 'gender', 'age_group',
            'severity_level', 'run_number', 'model', 'temperature',
            'timestamp', 'raw_response',
        ]}
        row.update({
            'diagnosis_label': '999', 'diagnosis_yes_no': 999,
            'severity_score': 999, 'impairment_score': 999,
            'treatment_text': '999',
            'treatment_score': 999.0, 'treatment_type_profile': 999,
            'minimizing_text': '999',
            'minimizing_score': 999.0,
            'diagnostic_evidence': '999',
            'framing_label': '999',
            'gendered_count': 999, 'medicalized_count': 999,
            'distanced_count': 999, 'neutral_count': 999,
        })
        return row

    diag_label      = _extract_field(raw, 'Diagnosis')
    severity_text   = _extract_field(raw, 'Severity')
    impairment_text = _extract_field(raw, 'Impairment')
    treatment_text  = _extract_field(raw, 'Treatment Recommendations')
    evidence_text   = _extract_field(raw, 'Diagnostic Evidence')
    minimizing_text = _extract_minimizing_region(raw)

    diag_yes_no = 1 if _DEPRESSION_RE.search(diag_label) else 0
    if not diag_label:
        diag_label  = '999'
        diag_yes_no = 999

    framing = _score_framing(raw)

    row = {
        'variant_id':      record.get('variant_id'),
        'prompt_type':     record.get('prompt_type'),
        'gender':          record.get('gender'),
        'age_group':       record.get('age_group'),
        'severity_level':  record.get('severity_level'),
        'run_number':      record.get('run_number'),
        'model':           record.get('model'),
        'temperature':     record.get('temperature'),
        'timestamp':       record.get('timestamp'),
        'raw_response':    raw,
        'diagnosis_label': diag_label,
        'diagnosis_yes_no': diag_yes_no,
        'severity_score':   _parse_score(severity_text),
        'impairment_score': _parse_score(impairment_text),
        'treatment_text':   treatment_text if treatment_text else '999',
        'treatment_score':  999.0,
        'treatment_type_profile': 999,
        'minimizing_text':  minimizing_text if minimizing_text else '999',
        'minimizing_score': 999.0,
        'diagnostic_evidence': evidence_text if evidence_text else '999',
        **framing,
    }
    return row


def run(raw_dir: Path | None = None) -> pd.DataFrame:
    """Parse all JSON files in raw_dir; return DataFrame (includes treatment_text, minimizing_text)."""
    if raw_dir is None:
        raw_dir = _RAW_DIR

    files = sorted(raw_dir.glob('*.json'))
    print(f"=== parser.py START ===  {len(files)} files in {raw_dir}/", flush=True)

    rows    = []
    skipped = 0
    for path in files:
        row = parse_file(path)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    print(
        f"=== parser.py DONE  ===  {len(rows)} rows parsed  ({skipped} skipped)",
        flush=True,
    )
    return pd.DataFrame(rows)
