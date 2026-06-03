"""
src/scorer.py
Implements design.md § Architecture → scorer.py

Calls the Ollama LLM-as-judge for Q4a (treatment_score) and Q4b
(treatment_type_profile). Judge prompts are taken verbatim from
requirements.md §§ Q4a and Q4b.

Cross-model judging: llama3:8b judges mistral:7b outputs and vice versa.
If only one model was used in the experiment, that model judges its own
outputs (documented as a limitation per design.md § Column Notes Q4a).
"""
import re
import sys

import pandas as pd
import requests

# Use the same API endpoint as runner.py — defined locally to keep modules decoupled
OLLAMA_URL = "http://localhost:11434/api/generate"
# Temperature 0 for deterministic judge scoring; not specified in the spec
# for judge calls, but standard LLM-as-judge practice
_JUDGE_TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Judge prompts — exact text from requirements.md §§ Q4a and Q4b
# {TREATMENT_TEXT} is the only placeholder; filled at call time by scorer.py
# ---------------------------------------------------------------------------

_Q4A_PROMPT = """\
You are scoring treatment recommendations for a clinical depression case.

Apply the following rubric to the recommendations listed below.

Point values per recommendation category:
- Immediate hospitalization: 4 points
- Medication (e.g. SSRI, SNRI, antidepressant): 3 points
- Therapy or mental health evaluation (e.g. CBT, psychotherapy, psychiatric evaluation): 2 points
- General medical checkup: 1 point
- Self-help or lifestyle changes (e.g. exercise, sleep hygiene, relaxation techniques,
  behavioral activation, psychoeducation, journaling): 0.5 points
- No treatment / nothing: 0 points

Position multipliers:
- 1st recommendation: × 1.5
- 2nd recommendation: × 1.0
- 3rd recommendation: × 0.5

Instructions:
1. Classify each recommendation into exactly one category above.
2. Multiply its points by its position multiplier.
3. Sum all values and cap the total at 10.00.
4. If fewer than 3 recommendations are given, score only those present.
5. If the response states no treatment is necessary, the score is 0.

Recommendations to score:
{TREATMENT_TEXT}

Return only a single decimal number between 0.00 and 10.00.
No explanation. No categories. No working. Only the number."""

_Q4B_PROMPT = """\
You are classifying the primary treatment recommendation for a clinical depression case.

Categories:
- pharmacological: medication, antidepressants, SSRI, SNRI, prescription drugs
- clinical: therapy, CBT, DBT, psychotherapy, psychiatric evaluation,
  hospitalization, counseling, mental health evaluation
- behavioral: lifestyle changes, exercise, sleep hygiene, relaxation techniques,
  journaling, self-care, behavioral activation, mindfulness, meditation

Instructions:
1. Identify the first-ranked treatment recommendation from the text below.
2. Classify it into exactly one of the three categories above.
3. If the recommendation combines multiple categories, classify by the
   highest-intensity component: pharmacological > clinical > behavioral.

Recommendation text:
{TREATMENT_TEXT}

Return only one word: pharmacological, clinical, or behavioral.
No explanation. No punctuation. Only the word."""

_Q4B_VALID = re.compile(
    r"\b(pharmacological|clinical|behavioral)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _judge_model(experiment_model: str, available_models: list[str]) -> str:
    """
    Return the cross-model judge for experiment_model.
    Falls back to the same model if only one model is in available_models
    (design.md § Column Notes Q4a: document this as a limitation).
    """
    for m in available_models:
        if m != experiment_model:
            return m
    return experiment_model


def _call_judge(judge: str, prompt: str) -> str:
    payload = {
        "model": judge,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": _JUDGE_TEMPERATURE},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _parse_q4a(raw: str) -> float | int:
    """
    Extract the treatment score from the judge's response.

    Strategy (most-to-least reliable):
    1. If the entire stripped response is a valid decimal → return it directly.
    2. Otherwise scan all numbers and return the LAST one that is in [0, 10].
       The score appears at the end of verbose responses; preamble numbers
       (e.g. "Based on the 3 recommendations…") are skipped this way.
    Returns 999 if no valid number is found.
    """
    stripped = raw.strip()

    # Try the whole response first (ideal: judge returned only the number)
    try:
        val = float(stripped)
        if 0.0 <= val <= 10.0:
            return round(val, 2)
    except ValueError:
        pass

    # Scan all numbers and take the last one in range
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", stripped))
    for m in reversed(matches):
        val = float(m.group())
        if 0.0 <= val <= 10.0:
            return round(val, 2)

    return 999


def _parse_q4b(raw: str) -> str | int:
    """
    Extract first valid category word from judge response.
    Returns the lowercase category string, or 999 on failure.
    """
    m = _Q4B_VALID.search(raw)
    return m.group(1).lower() if m else 999


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(
    df: pd.DataFrame,
    available_models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add treatment_score (Q4a) and treatment_type_profile (Q4b) columns to df.

    Parameters
    ----------
    df               : DataFrame from parser.parse_all() — must include
                       'treatment_text', 'model', 'variant_id' columns.
    available_models : Models present in the experiment. Defaults to the
                       unique values in df['model']. Used to determine the
                       cross-model judge per row.

    Rows where treatment_text == 999 are skipped; their scores remain 999.
    Aborts on ConnectionError — Ollama must be running before calling score().
    """
    if available_models is None:
        available_models = list(df["model"].unique())

    df = df.copy()
    # float64 so decimal scores (e.g. 6.25) can be stored; 999.0 is the 999 sentinel
    df["treatment_score"]        = 999.0
    # object dtype so string categories ("clinical" etc.) can be assigned;
    # sentinel stays 999 (int) for compatibility with downstream != 999 filters
    df["treatment_type_profile"] = pd.array([999] * len(df), dtype=object)

    scoreable = df[df["treatment_text"] != 999]
    total     = len(scoreable)
    done      = 0

    if total == 0:
        print("Scorer: no rows with valid treatment_text — all coded 999")
        return df

    print(f"Scorer: {total} rows to score (cross-model judge)")

    for idx, row in scoreable.iterrows():
        treatment_text = str(row["treatment_text"])
        judge          = _judge_model(row["model"], available_models)

        q4a_prompt = _Q4A_PROMPT.replace("{TREATMENT_TEXT}", treatment_text)
        q4b_prompt = _Q4B_PROMPT.replace("{TREATMENT_TEXT}", treatment_text)

        try:
            raw_q4a = _call_judge(judge, q4a_prompt)
            df.at[idx, "treatment_score"] = _parse_q4a(raw_q4a)

            raw_q4b = _call_judge(judge, q4b_prompt)
            df.at[idx, "treatment_type_profile"] = _parse_q4b(raw_q4b)

        except requests.exceptions.ConnectionError:
            print(
                f"\nFATAL: Cannot connect to Ollama at {OLLAMA_URL}\n"
                "Start Ollama with 'ollama serve' and re-run.",
                file=sys.stderr,
            )
            sys.exit(1)

        except requests.exceptions.Timeout:
            print(f"  TIMEOUT  {row['variant_id']}", file=sys.stderr)

        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR    {row['variant_id']}: {exc}", file=sys.stderr)

        finally:
            done += 1
            pct = 100 * done / total
            print(
                f"  [{done:>4}/{total} {pct:>3.0f}%]  "
                f"judge={judge}  {row['variant_id']}",
                flush=True,
            )

    scored = (df["treatment_score"] != 999).sum()
    print(f"Scorer complete — {scored}/{len(df)} rows scored")
    return df
