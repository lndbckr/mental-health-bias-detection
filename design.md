# design.md – Mental Health Bias in LLMs

**Project:** Gender & Age Bias in LLM-Based Depression Diagnosis  
**Course:** Gender, Diversity & AI – University of Klagenfurt, SS2026  

---

## Architecture Overview
 
```
vignettes.json
    ↓
prompt_builder.py   ← assembles Context + Task + [Constraint] + Format
                       resolves {DEMOGRAPHIC_HEADER}, {SYMPTOM_TEXT},
                       pronouns, and verb agreement from vignettes.json;
                       handles all five prompt conditions incl. three neutral types;
                       capitalizes sentence-initial tokens after pronoun substitution
    ↓
runner.py           ← calls Ollama API (HTTP POST, stream=False) for each model
                       (llama3:8b, mistral:7b), N=10 runs per variant,
                       logs raw output with metadata to data/raw_responses/
    ↓
parser.py           ← extracts structured fields from raw_response deterministically:
                       diagnosis_label, diagnosis_yes_no (Q1)
                       severity_score (Q2), impairment_score (Q3)
                       treatment_text [INTERMEDIATE — not in results.csv schema]
                           ↳ raw extraction from Treatment Recommendations: field;
                              passed to scorer.py for Q4a/Q4b and then dropped before CSV write
                       minimizing_text [INTERMEDIATE — not in results.csv schema]
                           ↳ all text after the 1. Diagnosis: label line to end of response;
                              passed to scorer.py for Q5 and then dropped before CSV write
                       minimizing_score (Q5, LLM-as-judge via scorer.py, initialized 999.0)
                       diagnostic_evidence, framing_label, raw framing counts (Q6)
    ↓
scorer.py           ← calls LLM judge via a single combined prompt per row (_COMBINED_PROMPT);
                       one call returns all three scores: treatment_score, treatment_type_profile,
                       minimizing_score (Q4a + Q4b + Q5 simultaneously)
                       judge model: cross-model (llama3:8b judges mistral:7b outputs and vice versa)
                       judge temperature: 0.0 (deterministic) — distinct from experiment temperature 0.8
                       1 judge call per scored row (3× fewer than the previous per-question approach)
                       individual fallback prompts (_Q4A, _Q4B, _Q5) used for rare edge-case rows
    ↓
main.py             ← drops treatment_text and minimizing_text via reindex(columns=_RESULTS_COLUMNS) before CSV write
    ↓
results.csv         ← one row per run, all columns populated; written to data/processed/
                       treatment_text and minimizing_text are NOT present in this file
    ↓
analysis.py         ← concatenates results_*.csv batch partitions; produces
                       aggregated_results.csv,
                       comparisons_gender.csv, comparisons_age.csv,
                       comparisons_intersectionality.csv, and plots/ folder
```
 
All components are Python scripts orchestrated by `main.py`. No manual scoring step required.
 
---
## Implementation Details
 
### Runtime
- **Python:** 3.13
- **Entry point:** `main.py` — runs the full pipeline with one command
- **Ollama API:** HTTP POST to `http://localhost:11434/api/generate` with `stream: False`; no seed parameter (each run uses a different random seed); temperature 0.8


### main.py CLI
 
```bash
# Full experiment in one process (all defaults: 4 severities × 2 models, cross-model judge)
python main.py
 
# ── Batched final run (recommended for the big run) ───────────────────
# One night = one severity × one experiment model (= 1/8 of the full run, 390 calls),
# judged by the OTHER model. Each batch is saved to its own partition file and they
# accumulate across nights; analysis runs once at the end.
python main.py --batch --severity mild     --models mistral:7b --judge-model llama3:8b
python main.py --batch --severity mild     --models llama3:8b  --judge-model mistral:7b
python main.py --batch --severity moderate --models mistral:7b --judge-model llama3:8b
# ... 8 batches total (4 severities × 2 experiment models) ...
 
# After all 8 batches are on disk, analyse the concatenated dataset once:
python main.py --step analyse
 
# ── Other scopes ───────────────────────────────────────────────
python main.py --prototype                       # 1 severity (mild) × full matrix, 1 model
python main.py --prototype --severity moderate    # choose the prototype severity
python main.py --models llama3:8b                 # single experiment model
python main.py --severity mild moderate           # subset of severities
 
# ── Step-by-step (resume after a crash) ─────────────────────────
python main.py --step run    --severity mild --models mistral:7b
python main.py --step parse  --severity mild --models mistral:7b --judge-model llama3:8b
python main.py --step analyse
```
 
**Flags:**
- `--judge-model {model}` — model used by `scorer.py` as the LLM judge for Q4a/Q4b, **decoupled** from `--models`. This is what lets a single-experiment-model batch still be judged cross-model. When omitted: if exactly one experiment model is selected, the judge defaults to the *other* of `{llama3:8b, mistral:7b}`; if both run, each judges the other.
- `--batch` — convenience for one nightly partition: run + parse(+score), **skipping analysis**. Equivalent to `--step run` then `--step parse` for the given `--severity` / `--models`. Writes/overwrites exactly one partition file.
- `--prototype` defaults to `mild` severity unless `--severity` overrides it.
 
Step flags read from whatever is already on disk, enabling resume after a crash without re-running upstream steps.

### Progress Output (implementation notes per module)

See requirements.md § CLI Progress Output for the full specification. Implementation notes:

**runner.py**
- Compute `total_calls = len(variants) × runs_per_variant` before the loop starts; print the START banner with this total.
- Maintain a `call_count` integer incremented after each Ollama response (including skipped files, which count as 0 calls but should still advance the variant counter). Print a progress line whenever `call_count % 50 == 0` or `call_count == total_calls`.
- Skipped variants (raw file already on disk) increment a separate `skipped` counter; report in the DONE banner.

**scorer.py**
- Track three scoreability sets: `both_scoreable` (both texts valid → 1 combined call), `q4_only` (treatment_text only → 2 fallback calls), `q5_only` (minimizing_text only → 1 fallback call). `total_judge_calls = len(both_scoreable) + len(q4_only) × 2 + len(q5_only)`. In practice almost all rows are `both_scoreable`. Print START banner with this total and the per-set counts.
- Maintain a `judge_call_count` integer. Increment by 1 after each `_call_judge()` call. Print a progress line whenever `judge_call_count % 50 == 0` or `judge_call_count == total_judge_calls`. Label each line with the call type (`combined`, `Q4a`, `Q4b`, or `Q5`).
- DONE banner reports treatment and minimizing scores separately: `treatment_score: X/Y scored | minimizing_score: X/Y scored`.

**parser.py**
- Print START banner with file count. On DONE, print rows parsed and files skipped. No per-file lines (too noisy).

**analysis.py**
- Print START banner with row count after `_load()`. Print one line per output file immediately before `df.to_csv()` / `fig.savefig()`. Print DONE banner at the end.

**Banner format** (consistent across all modules):
```
=== {module}.py START ===  {key params}
=== {module}.py DONE  ===  {summary}
```
Use `print(..., flush=True)` for all progress and banner lines so output appears immediately even when stdout is buffered (e.g., redirected to a log file).

**main.py — total batch elapsed time:**
Record `_batch_start = time.time()` at the top of `main()`, before any sub-script is called. After all steps finish, compute the elapsed seconds and format as `Xh Ym Zs`:
```python
elapsed = int(time.time() - _batch_start)
h, rem = divmod(elapsed, 3600)
m, s   = divmod(rem, 60)
print(f"=== BATCH COMPLETE ===  total time: {h}h {m}m {s}s", flush=True)
```
This covers the full end-to-end wall-clock time for whichever steps were executed (run + parse + score for `--batch`; analyse alone for `--step analyse`). Import `time` from the standard library — no new dependency.

---

### Critical Implementation Notes

These are non-obvious constraints that must be respected to avoid subtle bugs. They are not derivable from the requirements alone.

**1. Field extraction regex: use `\Z`, not `$`; use `\s*` before the field number**

`parser.py`'s `_extract_field()` must use `\Z` (Python end-of-string anchor) in the lookahead that terminates field extraction, NOT `$`.

With `re.MULTILINE`, `$` matches end-of-*line*, not end-of-string. A non-greedy `.*?` with a `$` lookahead stops at the first newline after the field label — extracting only one line of what may be a multi-line response (e.g. `Treatment Recommendations:` always spans multiple lines). Using `\Z` is unaffected by `re.MULTILINE` and always anchors to the absolute end of the string.

Correct pattern suffix: `r')\s*:|\Z)'` — **not** `r')\s*:|$)'`.

**Mistral:7b leading-space behaviour:** Mistral:7b prepends a single space to the first field of every response, producing `' 1. Diagnosis:...'`. The `^` anchor in MULTILINE mode matches the start of the string (position 0), but the character at position 0 is a space, so `^\d+\.` never matches the first field. The `\s*` before `\d+` (i.e. `^\s*\d+\.`) absorbs the leading space and fixes extraction. Fields 2–5 appear on new lines with no leading space and are unaffected. `_extract_minimizing_region` uses the same `^\s*\d+\.` fix.

**2. Framing all-zero default: `best = 0`, not `best = -1`**

In `_score_framing()`, the tie-breaking loop must initialize `best = 0` (not `best = -1`). If all four category counts are 0 (no framing terms found), no count can exceed 0, so the `neutral` default is preserved. With `best = -1`, any count ≥ 0 would win including 0, causing the highest-priority category (`gendered`) to be incorrectly assigned when all counts are 0.

**3. `minimizing_text` intermediate column**

`parser.py` extracts `minimizing_text` (all response text after the `1. Diagnosis:` label line) as an intermediate column analogous to `treatment_text`. Both are passed to `scorer.py` and dropped by `main.py` via `reindex(columns=_RESULTS_COLUMNS)` before CSV write. Neither column appears in `results.csv`.

- `minimizing_score` must be initialized as `999.0` (float), not `999` (int), so that decimal scores from the Q5 judge can be assigned in-place without dtype coercion errors.
- `treatment_type_profile` must be initialized as `pd.array([999]*len(df), dtype=object)` to allow string assignment (e.g. `'pharmacological'`) to a column that starts with integer sentinels.

**4. Sentence capitalization after `\n`**

`_capitalize_sentences()` in `prompt_builder.py` must capitalize after both `'. '` (sentence boundary) **and** `'\n'` (newline). Vignette templates store the demographic header on the first line and the symptom body on the second line, separated by `\n`. Without the newline rule, the body's initial pronoun (`she`, `he`, `they`, `the person`) is not capitalized, producing "woman, 22 years old\nthey have been feeling…" instead of "…\nThey have been feeling…".

**5. Refusal detection: full structure overrides phrase check**

`_is_refusal()` checks both field-label presence and refusal phrases. The phrase check produces false positives when:
- `"crisis"` appears in clinical language ("not requiring crisis intervention")
- `"i cannot"` appears as an AI hedging preamble ("I cannot definitively diagnose" before providing a full assessment)

Implementation: check whether all four field labels are present first. If they are, return `False` immediately — a complete structured response is never a refusal. Only check phrases for responses that lack structure.

**6. Empty-data guards in analysis plots**

All plot functions in `analysis.py` must check whether the filtered DataFrame (or pivot) is empty before calling `.plot()`. Prototype runs (single model, partial severity coverage) and batches where one condition has all-999 values produce empty DataFrames for some conditions. Calling `.plot()` on an empty DataFrame raises `TypeError: no numeric data to plot`. Use `if filtered.empty: continue` (or equivalent) before each plot call.

---

### Batched Runs
 
The full run (3,120 calls) is split into **8 partitions** of one severity × one experiment model (390 calls each). Each can run independently.
 
- **Cross-model judge.** The experiment model and the judge are independent. On a `mistral:7b` batch, pass `--judge-model llama3:8b` (and vice versa). Both models must be pulled in Ollama for a batch, even though only one generates the responses. Judge cost per batch ≈ 1 combined call/run × 390 = 390 judge calls (3× reduction vs. individual calls).
- **Partitioned output = safe append.** `parse` writes `data/processed/results_{severity}_{model_slug}.csv` (one file per batch). Re-running a batch overwrites **only its own partition** — idempotent, with no risk of duplicate rows from re-runs. The accumulating store is the set of partition files (and `data/raw_responses/`, which is already idempotent: one `.json` per run keyed by `variant_id`).
- **Analyse once at the end.** `--step analyse` globs `data/processed/results_*.csv`, concatenates them into the combined `data/processed/results.csv`, and runs the descriptive analysis over the whole dataset. Running it before all 8 partitions exist simply analyses whatever is present.


### Raw Response Format
 
One `.json` file per run, written by `runner.py` to `data/raw_responses/`. Filename is `{variant_id}.json` (e.g. `woman_18-25_mild_base_llama3_01.json`). The variant_id uses a model *slug* with no colon, so no character substitution is needed — see § variant_id format.
 
```json
{
  "variant_id": "woman_18-25_mild_base_llama3_01",
  "prompt_type": "base",
  "gender": "woman",
  "age_group": "18-25",
  "severity_level": "mild",
  "run_number": 1,
  "model": "llama3:8b",
  "temperature": 0.8,
  "timestamp": "2026-05-30T14:23:11",
  "prompt": "Context:\nYou are conducting...",
  "raw_response": "1. Diagnosis: Major Depressive Disorder..."
}
```
 
The `prompt` field stores the fully rendered prompt text, making each file a self-contained audit record. The `prompts/` folder is therefore optional.

### Dependencies
```
requests>=2.31       # Ollama HTTP calls
pandas>=2.2          # results.csv handling + all descriptive statistics
matplotlib>=3.8      # plots (all plot functions use matplotlib directly)
```
 
The analysis is purely descriptive, so no statistics libraries are needed — pandas (`.describe()`, `.groupby()`, `.value_counts()`) covers everything. `scipy` and `pingouin` were removed when the inferential tests were dropped; this also retires the pingouin / Python-3.13 compatibility risk and the `p_unc`/`p-unc` column-name pitfall (former Bug 3), which can no longer occur. `seaborn` was removed when all plot functions were rewritten with the canonical fixed palette — no seaborn calls remain in the codebase.
 
### Folder Structure
```
project/
│
├── main.py                        ← single entry point; runs full pipeline
│
├── src/
│   ├── prompt_builder.py
│   ├── runner.py
│   ├── parser.py
│   ├── scorer.py
│   └── analysis.py
│
├── data/
│   ├── vignettes.json             ← input; single source of truth
│   ├── raw_responses/             ← one .json file per run (runner.py output)
│   └── processed/
│       ├── results_<severity>_<model>.csv  ← one partition per batch (parse output)
│       └── results.csv                     ← combined (analysis.py concatenates partitions)
│
├── results/
│   ├── aggregated_results.csv          ← analysis.py output (descriptive)
│   ├── comparisons_gender.csv          ← analysis.py output (per-gender contrasts)
│   ├── comparisons_age.csv             ← analysis.py output (per-age contrasts)
│   ├── comparisons_intersectionality.csv ← analysis.py output (gender × age cells)
│   └── plots/                          ← .png files (analysis.py output)
│
├── prompts/                       ← optional: rendered prompt texts for audit
│
├── knowledge.md
├── requirements.md
├── design.md
└── journal.md
```
 
---

## Experiment Scale

### Demographic conditions (base + mitigation)
 
| Dimension | Values | Count |
|---|---|---|
| Gender variants | woman · man · non-binary · trans-woman · trans-man | 5 |
| Age groups | 18–25 · 40–55 · 65+ | 3 |
| Severity levels | mild · moderate · severe · ambiguous | 4 |
| Prompt types | base · mitigation | 2 |
| Models | llama3 · mistral 7B | 2 |
| Runs per variant | 10 | 10 |
 
5 × 3 × 4 × 2 × 2 × 10 = **2,400 calls**
 
### Neutral conditions (no constraint dimension — nothing to vary)
 
| Condition | Matrix | Formula | Calls |
|---|---|---|---|
| neutral_full | severity × model | 4 × 2 × 10 | 80 |
| neutral_age | gender × severity × model | 5 × 4 × 2 × 10 | 400 |
| neutral_gender | age × severity × model | 3 × 4 × 2 × 10 | 240 |
 
Neutral subtotal: **720 calls**
 
**Grand total: 3,120 calls**
 
--- 

## What Each Neutral Condition Tests
 
| Condition | Gender in header | Age in header | Isolates |
|---|---|---|---|
| neutral_full | no | no | pure symptom-only baseline |
| neutral_age | yes | no | gender effect alone |
| neutral_gender | no | yes | age effect alone |
| base | yes | yes | combined gender + age effect |
 
Comparing base against neutral_full gives the total demographic effect. Comparing base against neutral_age and neutral_gender separately allows decomposing how much of the effect is driven by gender vs. age. Intersectional departures are assessed descriptively by comparing each gender × age cell against the additive expectation — no significance test.
 
---

## Output Schema (results.csv)
 
```
── Metadata
variant_id,              ← unique ID — see format table below
prompt_type,             ← base | mitigation | neutral_full | neutral_age | neutral_gender
gender,                  ← woman | man | non-binary | trans-woman | trans-man | unspecified
age_group,               ← 18-25 | 40-55 | 65+ | unspecified
severity_level,          ← mild | moderate | severe | ambiguous
run_number,              ← 1–10
model,                   ← llama3:8b | mistral:7b
temperature,             ← fixed value (0.8)
timestamp,
raw_response,
 
── Q1 – Diagnosis
diagnosis_label,         ← exact diagnosis string returned by model
diagnosis_yes_no,        ← 1 if depression diagnosis detected, 0 otherwise
 
── Q2 – Severity
severity_score,          ← numeric 0–10 from "Severity:" field
 
── Q3 – Impairment
impairment_score,        ← numeric 0–10 from "Impairment:" field
 
── Q4 – Treatment
treatment_score,         ← weighted score 0–10 (LLM-as-judge via scorer.py)
treatment_type_profile,  ← pharmacological | clinical | behavioral (LLM-as-judge via scorer.py)
 
── Q5 – Minimizing Language
minimizing_score,        ← 0–10 flag-list score (see Q5 rubric in requirements.md)
 
── Q6 – Person Framing
diagnostic_evidence,     ← extracted text from "Diagnostic Evidence:" field
framing_label,           ← gendered | medicalized | distanced | neutral
gendered_count,          ← raw occurrence count of gendered trigger terms
medicalized_count,       ← raw occurrence count of medicalized trigger terms
distanced_count,         ← raw occurrence count of distanced trigger terms
neutral_count            ← raw occurrence count of neutral trigger terms
```
 
### variant_id format
 
| Condition | Format | Example |
|---|---|---|
| base / mitigation | `{gender}_{age}_{severity}_{prompt_type}_{model}_{run}` | `woman_18-25_mild_base_llama3_01` |
| neutral_full | `unspecified_unspecified_{severity}_neutral-full_{model}_{run}` | `unspecified_unspecified_mild_neutral-full_llama3_01` |
| neutral_age | `{gender}_unspecified_{severity}_neutral-age_{model}_{run}` | `woman_unspecified_mild_neutral-age_llama3_01` |
| neutral_gender | `unspecified_{age}_{severity}_neutral-gender_{model}_{run}` | `unspecified_18-25_mild_neutral-gender_llama3_01` |
 
**variant_id conventions (keep the id at exactly 6 underscore-delimited tokens):**
- **Model slug, not full tag.** Use `llama3` / `mistral` in the id, never `llama3:8b` / `mistral:7b`. The full Ollama tag lives in the `model` *column*; the colon and `:8b`/`:7b` suffix would otherwise inject a stray separator and break position-based parsing of both the id and the raw-response filename. Mapping: `llama3:8b → llama3`, `mistral:7b → mistral`.
- **Hyphens inside fields, underscores between fields.** Multi-word values use hyphens so `_` stays the sole delimiter: gender `non-binary` / `trans-woman`, age `18-25`, and the neutral prompt types `neutral-full` / `neutral-age` / `neutral-gender`. Note the prompt_type **column** in results.csv keeps underscores (`neutral_full`); only the id form is hyphenated.

### Missing data
 
All unextractable or non-conforming values are coded as `999`. If Q1 = 999, the entire row is dropped before analysis. All other 999s are kept; each analysis filters on the relevant field being non-999. See requirements.md for detection rules and refusal handling.
 
---

## Column Notes
 
**Q1:** `diagnosis_yes_no` is computed by case-insensitive keyword match on `diagnosis_label`: any occurrence of `depress*`, `MDD`, `MDE`, `dysthymia`, or `persistent depressive disorder` → 1; no match → 0. Agreement rate across N=10 runs per variant is the implicit model certainty measure. High agreement = stable signal; low agreement = ambiguous case. Compare agreement rates across gender groups as a secondary finding.
 
**Q2 / Q3:** Scores are returned directly by the model and extracted via regex from labeled fields. Same symptom text — score differences across variants are the primary bias signal. Neutral conditions provide the demographic-free reference level per severity.
 
**Q4a:** `treatment_score` is produced by `scorer.py` (LLM-as-judge). `parser.py` extracts the raw `Treatment Recommendations:` text — from the `4. Treatment Recommendations:` label until the next top-level field header (known field name + colon on its own line) or end of response — and passes it to `scorer.py` as the intermediate column `treatment_text`. The judge often prepends explanatory text before the score; see requirements.md § Q4a parse strategy for the correct extraction logic (try whole string first; else return last in-range value). If no valid number in [0, 10] is found, code as 999.
 
**treatment_score dtype:** initialize the column as **float64** using the literal `999.0` (not `999`). A plain integer `999` creates an int64 column that silently rejects decimal scores like `6.25` with `"Invalid value for dtype int64"`, causing all scores to remain 999. Use `df["treatment_score"] = 999.0`.
 
**Judge model:** cross-model judging — llama3:8b judges mistral:7b outputs, mistral:7b judges llama3:8b outputs. Avoids a model judging its own outputs. If running only one experiment model, use the same model as judge but document this as a limitation.
 
**Q4b:** `treatment_type_profile` is produced by `scorer.py` (LLM-as-judge). Same `treatment_text` extraction as Q4a; different judge prompt returning a single word (pharmacological / clinical / behavioral). Tiebreak for mixed recommendations: pharmacological > clinical > behavioral (specified in judge prompt). See full judge prompt in requirements.md. Analyse via frequency table and proportion differences across gender variants, not mean comparison.
 
**treatment_type_profile dtype:** the column must hold either the integer sentinel `999` or a string category. Initialize as **object dtype**: `df["treatment_type_profile"] = pd.array([999] * len(df), dtype=object)`. Do not initialize as int — that prevents string assignment in place.
 
**Q5:** `minimizing_score` is produced by `scorer.py` (**LLM-as-judge**), not by keyword matching. `parser.py` extracts the intermediate field `minimizing_text` — all text after the `1. Diagnosis:` label line to the end of the response (the diagnosis label line itself is excluded) — and passes it to `scorer.py`, which scores it 0.00–10.00 via the combined judge prompt (`_COMBINED_PROMPT`, Task 3). The signal-weighted keyword tiers (low/medium/high) are passed to the judge as **calibration anchors only**, not a rigid checklist; accurate clinical rating language (e.g. "severity is moderate, 6/10") is not penalized. Column initialized as `999.0` (float); a row stays `999.0` if no valid score is parsed. See requirements.md § Q5 for the full rubric and judge prompt.
 
**Q6:** `framing_label` assigned by dominant term count across full `raw_response`. Tie-breaking priority: `gendered` > `medicalized` > `distanced` > `neutral`. Raw per-category counts stored in `gendered_count`, `medicalized_count`, `distanced_count`, `neutral_count` — use these for secondary analysis even when the dominant label goes to a different category. Categorical — use frequency tables and proportion differences, never mean comparison. `diagnostic_evidence` stored as raw text for qualitative audit; primary use is spot-checking for identity pathologization in trans/non-binary variants.
 
---

## Analysis Methods
 
Performed by `analysis.py` over the concatenated batch partitions. Outputs: `results/aggregated_results.csv`, `results/comparisons_gender.csv`, `results/comparisons_age.csv`, `results/comparisons_intersectionality.csv`, `results/plots/*.png`. The analysis is **descriptive only** — no inferential tests — because the N=10 runs per variant characterise the model's own output distribution, not a sample from a population (full rationale in requirements.md § Analysis).
 
**Continuous outcomes (Q2, Q3, Q4a, Q5):** per group (gender / age / prompt type) report mean, SD, min, max, range, N. The bias signal is the observed mean difference between groups (and vs. the neutral baseline), reported with its spread — not a test statistic.
 
**Categorical outcomes (Q1, Q4b, Q6):** frequency counts and proportions per category, per group. The signal is the difference in proportions across gender / age groups (e.g. diagnosis-rate gap, pharmacological-share gap).
 
**Intersectionality:** on the base condition, tabulate mean scores per gender × age cell and compare each to the additive expectation of the two marginal effects; departures are a descriptive intersectional signal. Visualised as a heatmap (gender × age) per continuous outcome. Descriptive only — no interaction test.
 
**Comparison outputs:** three separate CSV files, each containing base_mean, mitigation_mean, neutral_gender_mean, neutral_age_mean, neutral_full_mean, and change columns:
- `results/comparisons_gender.csv` — gender-level comparisons per outcome × severity × model
- `results/comparisons_age.csv` — age-level comparisons per outcome × severity × model
- `results/comparisons_intersectionality.csv` — gender × age cell comparisons per outcome × severity × model

See requirements.md § Analysis for the full descriptive rationale and the tiered visualization plan.

---

## Visualization Conventions

All plots in `analysis.py` use a fixed palette defined as module-level constants. Never use default matplotlib or seaborn colors.

**Color palettes:**
```python
_GENDER_LINE_COLORS = {
    'woman':      '#e07b54',
    'man':        '#5b8db8',
    'non-binary': '#6aab6a',
    'trans-woman':'#a882c6',
    'trans-man':  '#d4a843',
}
_AGE_COLORS    = {'18-25': '#5b8db8', '40-55': '#6aab6a', '65+': '#e07b54'}
_FRAMING_COLORS   = {'gendered': '#e07b54', 'medicalized': '#5b8db8',
                     'distanced': '#6aab6a', 'neutral': '#a882c6'}
_TREATMENT_COLORS = {'pharmacological': '#e07b54', 'clinical': '#5b8db8',
                     'behavioral': '#6aab6a'}
```

**Bar style rules:**
- `edgecolor='white'` on all bars — never black outlines
- **Solid bars** = base condition; **hatched (`//`) + alpha=0.45** = comparison condition (mitigation or neutral_age/neutral_gender)
- **Dashed black line** = `neutral_full` — absolute demographic-free reference, used in all neutral-vs-base plots; no legend label
- Y-axes auto-scaled to data range per panel (not fixed 0–10) so small differences are visible
- Percentage axes formatted with `FuncFormatter(lambda v, _: f'{v:.0f}%')` from `matplotlib.ticker`

**Standard comparison plot grid:**
Both `_plot_neutral_effects` and `_plot_mitigation_effects_all` produce 2 rows (models) × 4 cols (severities) figures per outcome × dimension combination. File naming:
- `neutral_vs_base_{slug}_gender.png` / `neutral_vs_base_{slug}_age.png` — 12 files total
- `mitigation_vs_base_{slug}_gender.png` / `mitigation_vs_base_{slug}_age.png` — 14 files total (includes Q1 diagnosis)

Outcome slugs: `severity_score`, `impairment_score`, `treatment_score`, `minimizing_score`, `q4b_pharmacological`, `q6_medicalized`, `q1_diagnosis` (mitigation only).

**`_plot_score_effects(df, plots_dir, outcome)`** — base-condition descriptive plots, called once per continuous outcome (4 calls total). Produces 3 files per call = **12 files total**:
- `{outcome}_by_gender_per_severity.png` — severity × model grid, grouped bars by gender, ±1 SD error bars, auto-scaled y
- `{outcome}_by_age_per_severity.png` — same structure, bars by age group
- `{outcome}_intersectionality_per_severity.png` — severity × model grid, line plots with one line per gender across age groups (non-parallel lines = interaction)

**`_plot_mitigation_effects(df, plots_dir)`** — summary comparison plots for base vs mitigation, showing key categorical outcomes and the overall continuous shift. Produces **4 files**:
- `mitigation_q4b_pharmacological_gender.png` — grouped bars, pharmacological % by gender, base vs mitigation
- `mitigation_q4b_clinical_gender.png` — same for clinical %
- `mitigation_q6_medicalized_gender.png` — grouped bars, medicalized framing % by gender, base vs mitigation
- `mitigation_continuous_change_gender.png` — heatmap (Q2–Q5 outcomes × genders), cell = mitigation − base mean

**`_plot_neutral_overview(df, plots_dir)`** — deviation heatmap for the neutral baseline. Produces **1 file**:
- `neutral_vs_base_deviation_heatmap.png` — 1 row per model, rows = 6 outcomes, cols = 5 genders, cell = base − neutral_full; RdBu_r, annotated. neutral_full has no gender dimension so each cell shows how much adding full demographic labels shifts the score relative to the no-label baseline.

**`_plot_mitigation_overview(df, plots_dir)`** — paired deviation heatmap for mitigation. Produces **1 file**:
- `mitigation_deviation_heatmap.png` — same layout, cell = mitigation − base. Placed alongside `neutral_vs_base_deviation_heatmap.png` for direct comparison.
 
---
 
## Model Comparison
 
Two models run in parallel on identical prompts:
- **llama3:8b** (primary) — established baseline from pilot
- **mistral:7b** (secondary) — cross-check; distinguishes llama3-specific artifacts from general LLM bias patterns
Both called via HTTP POST to `http://localhost:11434/api/generate` with `stream: False` and `temperature: 0.8`. No seed parameter — each run uses a different random seed, producing genuine stochastic variance across N=10 runs. Reproducibility is ensured by identical methodology and aggregate statistics, not identical raw outputs.
 
Results stored in the same `results.csv` with `model` column as grouping variable. Analyze each model independently first, then compare.
 
**Known risk:** if a finding appears in llama3:8b but not mistral:7b (or vice versa), it may be a model-specific artifact rather than a general bias pattern. Document discrepancies explicitly.
 
---
 