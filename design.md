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
                       handles all five prompt conditions incl. three neutral types
    ↓
runner.py           ← calls Ollama API (HTTP POST, stream=False) for each model
                       (llama3:8b, mistral:7b), N=10 runs per variant,
                       logs raw output with metadata to data/raw_responses/
    ↓
parser.py           ← extracts structured fields from raw_response deterministically:
                       diagnosis_label, diagnosis_yes_no (Q1)
                       severity_score (Q2), impairment_score (Q3)
                       treatment_text (Q4 raw extraction, passed to scorer.py)
                       minimizing_score (Q5, flag list, scan region: post-diagnosis line)
                       diagnostic_evidence, framing_label, raw framing counts (Q6)
    ↓
scorer.py           ← calls LLM judge for Q4a and Q4b;
                       Q4a: sends treatment_text + Q4a judge prompt → numeric treatment_score
                       Q4b: sends treatment_text + Q4b judge prompt → categorical treatment_type_profile
                       judge model: cross-model (llama3:8b judges mistral:7b outputs and vice versa)
    ↓
results.csv         ← one row per run, all columns populated; written to data/processed/
    ↓
analysis.py         ← loads results.csv; produces aggregated_results.csv,
                       statistical_results.csv, and plots/ folder
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
# Full experiment (all defaults)
python main.py
 
# Prototype scope — 1 severity level × full matrix (~780 calls)
python main.py --prototype
python main.py --prototype --severity moderate   # specify which severity
 
# Subset runs
python main.py --models llama3:8b               # single model
python main.py --severity mild moderate          # subset of severities
 
# Step-by-step (resume from checkpoint)
python main.py --step run       # runner.py only — generates raw_responses/
python main.py --step parse     # parser.py + scorer.py only — reads raw_responses/, writes results.csv
python main.py --step analyse   # analysis.py only — reads results.csv, writes results/
```
 
`--prototype` defaults to `mild` severity unless `--severity` overrides it. Step flags read from whatever is already on disk, enabling resume after a crash without re-running upstream steps.

### Raw Response Format
 
One `.json` file per run, written by `runner.py` to `data/raw_responses/`. Filename mirrors the variant_id with colons replaced by underscores (e.g. `woman_18-25_mild_base_llama3_8b_01.json`).
 
```json
{
  "variant_id": "woman_18-25_mild_base_llama3:8b_01",
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
pandas>=2.2          # results.csv handling
scipy>=1.13          # statistical tests (Welch t-test, chi-square, Fisher)
pingouin>=0.5        # Welch ANOVA, Games-Howell post-hoc
matplotlib>=3.8      # plots
seaborn>=0.13        # plot styling
```
 
Note: verify `pingouin` compatibility with Python 3.13 before committing (`pip install pingouin`). If installation fails, Welch ANOVA and Games-Howell can be implemented manually using `scipy`.
 
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
│       └── results.csv            ← parser.py + scorer.py output
│
├── results/
│   ├── aggregated_results.csv     ← analysis.py output
│   ├── statistical_results.csv    ← analysis.py output
│   └── plots/                     ← .png files (analysis.py output)
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
 
*Scope reduction option for 03.06 prototype:* run neutral_age and neutral_gender on llama3:8b only (saves 320 calls, total 2,800). Add mistral:7b for the 24.06 final.
 
--- 

## What Each Neutral Condition Tests
 
| Condition | Gender in header | Age in header | Isolates |
|---|---|---|---|
| neutral_full | no | no | pure symptom-only baseline |
| neutral_age | yes | no | gender effect alone |
| neutral_gender | no | yes | age effect alone |
| base | yes | yes | combined gender + age effect |
 
Comparing base against neutral_full gives the total demographic effect. Comparing base against neutral_age and neutral_gender separately allows decomposing how much of the effect is driven by gender vs. age. The two-way ANOVA interaction term (gender × age) in analysis.py tests whether intersectional effects exceed the sum of individual effects.
 
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

### Missing data
 
All unextractable or non-conforming values are coded as `999`. If Q1 = 999, the entire row is dropped before analysis. All other 999s are kept; each analysis filters on the relevant field being non-999. See requirements.md for detection rules and refusal handling.
 
---

## Column Notes
 
**Q1:** `diagnosis_yes_no` is computed by case-insensitive keyword match on `diagnosis_label`: any occurrence of `depress*`, `MDD`, `MDE`, `dysthymia`, or `persistent depressive disorder` → 1; no match → 0. Agreement rate across N=10 runs per variant is the implicit model certainty measure. High agreement = stable signal; low agreement = ambiguous case. Compare agreement rates across gender groups as a secondary finding.
 
**Q2 / Q3:** Scores are returned directly by the model and extracted via regex from labeled fields. Same symptom text — score differences across variants are the primary bias signal. Neutral conditions provide the demographic-free reference level per severity.
 
**Q4a:** `treatment_score` is produced by `scorer.py` (LLM-as-judge). `parser.py` extracts the raw `Treatment Recommendations:` text — from the `4. Treatment Recommendations:` label until `5.` or end of response, whichever comes first — and passes it to `scorer.py`, which calls the judge model with the Q4a judge prompt (see requirements.md). The judge returns a single decimal number parsed with a simple float extraction. If the judge returns a non-numeric response, code as 999.
 
**Judge model:** cross-model judging — llama3:8b judges mistral:7b outputs, mistral:7b judges llama3:8b outputs. Avoids a model judging its own outputs. If running only one experiment model, use the same model as judge but document this as a limitation.
 
**Q4b:** `treatment_type_profile` is produced by `scorer.py` (LLM-as-judge). Same `treatment_text` extraction as Q4a; different judge prompt returning a single word (pharmacological / clinical / behavioral). Tiebreak for mixed recommendations: pharmacological > clinical > behavioral (specified in judge prompt). See full judge prompt in requirements.md. Analyze via frequency table + chi-square across gender variants, not mean comparison.
 
**Q5:** `minimizing_score` scanned from all text after the `1. Diagnosis:` label line to the end of the response (the diagnosis label line itself is excluded). Case-insensitive. Longest-phrase-first matching to avoid double-counting sub-phrases within a single match (e.g. "normal for their age" matched as one 3-point phrase, not as individual words). Each distinct occurrence counted independently — repeat occurrences score their full point value each time.
 
**Q6:** `framing_label` assigned by dominant term count across full `raw_response`. Tie-breaking priority: `gendered` > `medicalized` > `distanced` > `neutral`. Raw per-category counts stored in `gendered_count`, `medicalized_count`, `distanced_count`, `neutral_count` — use these for secondary analysis even when the dominant label goes to a different category. Categorical — use frequency tables and chi-square, never mean comparison. `diagnostic_evidence` stored as raw text for qualitative audit; primary use is spot-checking for identity pathologization in trans/non-binary variants.
 
---

## Analysis Methods
 
Performed by `analysis.py`. Outputs: `results/aggregated_results.csv`, `results/statistical_results.csv`, `results/plots/*.png`.
 
**Continuous outcomes (Q2, Q3, Q4a, Q5):** Welch's ANOVA per outcome × grouping dimension (gender / age / prompt type). Games-Howell post-hoc where significant. Bonferroni correction across 12 primary tests. Welch's t-tests for neutral vs. base comparisons per severity level.
 
**Categorical outcomes (Q1, Q4b, Q6):** Chi-square or Fisher's exact test (if expected cell count < 5) across gender and age groups.
 
**Intersectionality:** Two-way Welch ANOVA on base condition with gender × age interaction term. Significant interaction = intersectional bias present. Visualized as interaction plot per continuous outcome.
 
See requirements.md for full statistical rationale and visualization list.

---
 
## Model Comparison
 
Two models run in parallel on identical prompts:
- **llama3:8b** (primary) — established baseline from pilot
- **mistral:7b** (secondary) — cross-check; distinguishes llama3-specific artifacts from general LLM bias patterns
Both called via HTTP POST to `http://localhost:11434/api/generate` with `stream: False` and `temperature: 0.8`. No seed parameter — each run uses a different random seed, producing genuine stochastic variance across N=10 runs. Reproducibility is ensured by identical methodology and aggregate statistics, not identical raw outputs.
 
Results stored in the same `results.csv` with `model` column as grouping variable. Analyze each model independently first, then compare.
 
**Known risk:** if a finding appears in llama3:8b but not mistral:7b (or vice versa), it may be a model-specific artifact rather than a general bias pattern. Document discrepancies explicitly.
 
---
 