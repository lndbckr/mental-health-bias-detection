# journal.md – Mental Health Bias in LLMs

**Project:** Gender & Age Bias in LLM-Based Depression Diagnosis  
**Course:** Gender, Diversity & AI – University of Klagenfurt, SS2026  

---

## Log

### 2026-04-22 – Task 1 Submitted
- Literature review completed covering Chansiri et al. (2024), Schnepper et al. (2025), Lawrence et al. (2024).

### 2026-04-27 – Session 3 Feedback
- Key takeaways from session:
  - The Diversity Instruction Paradox.
  - Sycophancy trap: rubric-based evaluation only; never ask "is this biased?"
  - Gemini blocks sensitive mental health prompts at API level (documented FAIR-SW-Bench finding) – plan for this.
  - N ≥ 10 runs per variant is the minimum standard.
  - Scoring anchors (0 / 5 / 10) must be worked examples, not just labels.

### 2026-05-07 – Experiment Design Drafted 
- Decision to focus on depression
- Finalized prompt schema.
- Finalized gender variants: woman, man, non-binary, trans-woman, trans-man.
- Finalized age groups: 18–25, 40–55, 65+.
- Defined Q1–Q4 scoring with weighted treatment rubric.
- Mild, moderate, severe vignettes drafted; ambiguous still TBD.
- Decision: use llama3 and mistral via Ollama for pilot (local, free, reproducible).

### 2026-05-08 – Promptotyping Docs Drafted (Task 2 Submission)
- All four .md files created 
- Decisions made:
  - Separated "language bias" (minimizing/dismissive words, flag-based) from "stereotype bias" (rubric 0–10) as distinct scoring dimensions.
  - Pilot scope set at 1 severity × 5 gender × 2 prompt types (base + mitigation) + 1 neutral = 11 calls

### 2026-05-10 - Scoring Metrics Finalized
- Removed Confidence Score due to anchoring
- Changed Treatment Score: 
  - Q4a: Treatment Score (weighted)
  - Added Q4b: Treatment Type Profile (categorical)
- Added Q5: Minimizing / Dismissive Language
- Added Q6: Person Framing Label

### 2026-05-11 - Finalizing Vignettes
- Added Ambigious Vignette
- Created Vignette Templates
- Updated Pronoun / Agreement Table for Vigentte Generation
- Added vignettes.json for vignette Generation

### 2026-05-13 - LLM as a Judge
- Added requirements for LLM as Judge
- used for Q4a and Q4b

### 2026-05-16 – Prompt Conditions
- Expanded from 3 to 5 prompt conditions: base, mitigation, neutral_full, neutral_age, neutral_gender.
- Neutral conditions designed to decompose gender and age effects independently.
- neutral_full and neutral_gender use "the person" as subject to avoid they/their being read as non-binary identity signal.

### 2026-05-19 – Architecture
- scorer.py added as distinct pipeline step handling Q4a and Q4b (cross-model judging: llama3:8b judges mistral:7b and vice versa).
- analysis.py added as final pipeline step producing aggregated_results.csv, statistical_results.csv, and plots/.
- main.py confirmed as single entry point for full pipeline.

### 2026-05-20 – Statistical Analysis
- Welch's ANOVA + Games-Howell post-hoc for continuous outcomes.
- Chi-square / Fisher's exact for categorical outcomes.
- Bonferroni correction across 12 primary tests.
- Intersectionality: two-way Welch ANOVA with gender × age interaction term.
- Neutral vs. base decomposition via Welch's t-tests per severity level.

### 2026-05-21 – Implementation & Missing Data
- Python 3.13, temperature 0.8, no fixed seed.
- Raw responses stored as one .json per run in data/raw_responses/.
- Field-level 999 coding. Drop row only if Q1 = 999. No response_valid column.
- diagnosis_yes_no detection rule: case-insensitive match for depress*, MDD, MDE, dysthymia, persistent depressive disorder.
- treatment_text extraction: from "4. Treatment Recommendations:" until the next **top-level field header** (a line matching a known field name followed by a colon) or end of response. Numbered sub-items inside the treatment list (e.g., "1. SSRI", "2. CBT") are not field boundaries and must not stop extraction. treatment_text is an intermediate column produced by parser.py for scorer.py — it is dropped before writing results.csv.

### 2026-05-28 – Prototype Implementation: Bugs Found and Fixed

Three bugs were found during the prototype run (mild severity, llama3:8b, N=390). All three were silent failures — no crash, just wrong values. Documented here so the spec can be updated and future implementations avoid repeating them.

**Bug 1 — treatment_score and treatment_type_profile coded 999 for 389/390 rows**
Root cause: initializing `df["treatment_score"] = 999` (integer literal) creates a pandas int64 column. When the judge returns a decimal score like `6.25`, the assignment raises `"Invalid value '6.25' for dtype 'int64'"` silently caught by the except block, leaving the sentinel. The one row that succeeded had a judge response of `"0"` — an integer that fit int64.
Fix: initialize as float64 (`df["treatment_score"] = 999.0`) and treatment_type_profile as object dtype (`pd.array([999]*len(df), dtype=object)`).
Lesson: sentinel initialization for columns that will receive decimals or mixed types must use the correct dtype from the start.

**Bug 2 — Q4a parse extracted preamble number instead of score**
Root cause: the judge occasionally prepends explanatory text ("Based on the 3 recommendations above, the score is 6.25"). Extracting the first numeric match returned `3.0`, not `6.25`.
Fix: try parsing the entire stripped response as a float first (ideal case: judge returned only the number); if that fails, scan all numeric tokens in reverse and return the last value in [0, 10]. The score appears at the end of verbose responses; preamble numbers are skipped.
Additional edge case: `\b` word boundary before a negative sign (`-1`) can match only the `1`. Use `(-?\d+(?:\.\d+)?)` without `\b` before the sign.
Lesson: judge prompts that say "return only the number" are not always followed. Parse defensively.

**Bug 3 — pingouin ANOVA column name mismatch**
Root cause: `pg.welch_anova()` returns a DataFrame with column `p_unc` (underscore) in current pingouin releases; older documentation showed `p-unc` (hyphen). Hard-coding `aov["p-unc"]` raised `KeyError: 'p-unc'` for all 9 Welch ANOVA tests, which were caught and logged as "pingouin error" in statistical_results.csv.
Fix: probe at runtime — `p_col = "p_unc" if "p_unc" in aov.columns else "p-unc"`.
Lesson: library column names can drift across versions. Never hard-code output column names from external libraries without a version check or runtime probe.

### 2026-06-05 – Docs revision: descriptive stats, batched runs
 
**Consistency fixes (pre-existing doc bugs).**
- `variant_id` model token was inconsistent across design.md: `llama3` in the format table, `llama3:8b` in the JSON example, `llama3_8b` in the filename example. The `:8b` / `_8b` forms inject a stray separator and break the 6-token parse. Standardised on a model **slug** (`llama3`, `mistral`) in the id; the full Ollama tag stays only in the `model` column. Filenames need no colon replacement now.
- Documented the implicit prompt_type convention: hyphenated in `variant_id` (`neutral-full`), underscored in the column (`neutral_full`). Re-checked the call-count arithmetic (390 per batch × 8 = 3,120; prototype = 1/8) — internally consistent.

**Statistics: inferential → descriptive.**
- Feedback: no population is sampled, so inferential statistics are the wrong tool. Removed Welch's ANOVA, Games-Howell, Bonferroni, chi-square / Fisher, Welch's t-tests, and the two-way interaction test.
- Replaced with descriptive statistics: mean/SD/min/max/range/N for continuous; counts/proportions for categorical; observed mean/proportion differences between conditions (`base − neutral`, `base − mitigation`) in `comparisons.csv`; intersectionality as a descriptive additive-expectation check.
- Dependency cleanup: dropped `scipy` and `pingouin`. Side effect: Bug 3 (pingouin `p_unc` / `p-unc` column drift) can no longer recur, and the Python-3.13 pingouin-compat risk is gone. `statistical_results.csv` → `comparisons.csv`.

**Batched final run.**
- New workflow: one severity × one experiment model per batch (390 calls = 1/8 of 3,120), judged by the other model. Added `--judge-model` to decouple the judge from the experiment model, and `--batch` (run + score, no analysis).
- Outputs partitioned as `data/processed/results_<severity>_<model>.csv` — re-running a batch overwrites only its own partition (idempotent, no duplicate rows). `--step analyse` concatenates all partitions and runs once after all 8 batches.

### 2026-06-06 – CLI Progress Output Requirements Added

Long-running sub-scripts (runner.py, scorer.py) previously gave no feedback during execution, making it impossible to tell from the terminal whether the process was making progress or had stalled.

**Specification added (requirements.md § CLI Progress Output):**
- All sub-scripts print a START banner (script name + key parameters) and a DONE banner (summary counts) to stdout.
- **runner.py** and **scorer.py** print a progress line after every 50 LLM calls using a fixed-width counter `[NNNN/TTTT calls]`. scorer.py counts Q4a and Q4b calls separately (2 per row, 780 judge calls per 390-row batch), so milestones land at 50, 100, 150, … judge calls.
- **parser.py** (no LLM calls, fast) prints only start/done banners plus a line for any skipped/failed file.
- **analysis.py** prints one line per output file as it is written so the user can see which plots are done.
- All progress output → stdout; errors and warnings → stderr. All lines use `flush=True` so output appears immediately even when stdout is redirected to a log file.

### 2026-06-08 – First Batch Bugs Fixed + Combined Judge Call

**First batch completed (mild severity, mistral:7b, N=390).** Two silent bugs discovered and fixed; one scoring improvement made.

**Bug: All 390 rows coded diagnosis_yes_no = 999 → no CSV written.**
Root cause: Mistral:7b prepends a leading space before `1. Diagnosis:` — the raw response starts with ` 1. Diagnosis:` not `1. Diagnosis:`. The regex `^\d+\.` in `_extract_field()` and `_extract_minimizing_region()` requires the digit to be at column 0, so it never matched field 1 in any Mistral response. Fields 2–5 were unaffected because they follow a `\n` where `^` anchors correctly.
Fix: changed `^\d+\.` to `^\s*\d+\.` in both functions, and added `\s*` to the field-boundary lookahead pattern (`\n\s*\d+\.`). Documented in DESIGN.md Critical Implementation Notes.

**Bug: Refusal detector false positives on 2 valid responses.**
Two rows with complete five-field structure were incorrectly marked as refusals: one contained `"crisis"` in the phrase "not requiring crisis intervention"; the other contained `"I cannot"` in the preamble "I cannot definitively diagnose" before giving a full assessment.
Fix: added a structure check before phrase scanning — if all four field headers (`severity:`, `impairment:`, `treatment recommendations:`, `diagnostic evidence:`) are present in the response, the row is treated as valid regardless of phrase content. Phrase scanning only applies to structurally incomplete responses. Documented in DESIGN.md Critical Implementation Notes.

**Q6 framing: removed `their` from neutral term list.**
`their` was almost exclusively possessive in clinical text (e.g. "their symptoms", "their treatment plan") — not a subject-reference. Counting possessive uses inflated neutral framing scores for every demographic variant. Removed from `_FRAMING['neutral']`; list is now `['the person', 'they', 'them']`.

**Combined LLM judge call: Q4a + Q4b + Q5 merged into one prompt per row.**
Previously scorer.py made three separate Ollama calls per scoreable row — one each for Q4a (treatment_score), Q4b (treatment_type_profile), and Q5 (minimizing_score) — totalling ~1,170 judge calls per 390-row batch.
Replaced with a single `_COMBINED_PROMPT` that returns all three values in one response using strictly labeled output lines:
```
TREATMENT_SCORE: [decimal 0.00–10.00]
TREATMENT_TYPE: [pharmacological | clinical | behavioral]
MINIMIZING_SCORE: [decimal 0.00–10.00]
```
`_parse_combined()` scans the response for the three prefix labels; any missing or unparseable field falls back to 999. Three-set routing handles edge cases: rows where both `treatment_text` and `minimizing_text` are valid get one combined call; rows with only `treatment_text` fall back to two individual calls (Q4a + Q4b); rows with only `minimizing_text` get one individual Q5 call. The individual prompt constants (`_Q4A_PROMPT`, `_Q4B_PROMPT`, `_Q5_PROMPT`) are retained as fallbacks. Net effect: ~390 judge calls per batch instead of ~1,170 — 3× reduction in judge overhead and wall-clock time per nightly batch.

### 2026-06-18 – Full Analysis Overhaul: Plots, Comparisons, Bug Fix, Descriptive Findings

**Full dataset now available** (all 8 batches completed: 4 severities × 2 models). Analysis step (`python main.py --step analyse`) runs over the complete 3,120-call dataset.

**Old plot functions removed from analysis.py.**
Eight functions using the default matplotlib/seaborn palette were replaced and then fully deleted:
`_plot_continuous_by_gender`, `_plot_continuous_by_age`, `_plot_diagnosis_rate`, `_plot_base_vs_mitigation`, `_stacked_bar_by_gender`, `_plot_treatment_type`, `_plot_intersectionality`, `_plot_neutral_vs_base`. Corresponding 38 old `.png` files were deleted from `results/plots/`. All remaining plot functions use the fixed canonical palette (`_GENDER_LINE_COLORS`, `_AGE_COLORS`, `_TREATMENT_COLORS`, `_FRAMING_COLORS`) with `edgecolor='white'` and no seaborn calls. The `seaborn` import was removed from `analysis.py`.

**Standardized visualization conventions established** (documented in DESIGN.md § Visualization Conventions):
- Solid bars = base; hatched (`//`) + alpha=0.45 = comparison condition
- Dashed black line = `neutral_full` (absolute demographic-free reference) — used consistently in all neutral-vs-base plots, no legend label
- Y-axes auto-scaled per panel to zoom into the data range

**`_plot_neutral_effects(df, plots_dir)` — new generalized function.**
Produces 2 rows (models) × 4 cols (severities) comparison figures for each of 6 outcomes × 2 dimensions (gender, age) = 12 plots:
- Gender panels: solid=base per gender, hatched=neutral_age per gender, dashed=neutral_full
- Age panels: solid=base per age, hatched=neutral_gender per age, dashed=neutral_full
- Outcomes: severity_score, impairment_score, treatment_score, minimizing_score, q4b_pharmacological (%), q6_medicalized (%)
- Files: `neutral_vs_base_{slug}_gender.png` and `neutral_vs_base_{slug}_age.png`

**`_plot_mitigation_effects_all(df, plots_dir)` — new function, same grid structure.**
14 plots covering all 7 outcomes × 2 dimensions (adds Q1 diagnosis to the neutral-effects list). No neutral reference line — the mitigation plots focus solely on base vs. mitigation. Files: `mitigation_vs_base_{slug}_gender.png` and `mitigation_vs_base_{slug}_age.png`. Q1 diagnosis uses `pd.to_numeric()` with `errors='coerce'` to handle the integer sentinel correctly for categorical comparison.

**`_plot_neutral_diagnosis_ambiguous(df, plots_dir)` — new focused plot.**
Dedicated 2×2 figure for Q1 diagnosis rate on the ambiguous vignette only. Rows=models, cols=gender/age dimension. Shows base vs. neutral_age vs. neutral_full (gender panel) and base vs. neutral_gender vs. neutral_full (age panel). File: `neutral_vs_base_q1_diagnosis_ambiguous.png`.

**Comparisons CSV split into three files.**
Previously a single `comparisons.csv`. Now:
- `results/comparisons_gender.csv` — per-gender contrasts per outcome × severity × model
- `results/comparisons_age.csv` — per-age contrasts per outcome × severity × model
- `results/comparisons_intersectionality.csv` — gender × age cell comparisons per outcome × severity × model

**Bug fix: Boolean Series index mismatch in `_comparisons_age()`.**
In `_comparisons_age()`, the loop over age groups created a sub-frame `sdf = df[df['age_group'] == age]` with a reset slice index. Three filter masks then mixed `sdf` conditions with `df['gender'].isin(...)`, which has the original index — pandas re-indexed the boolean Series, sometimes silently picking wrong rows and always raising `UserWarning: Boolean Series key will be reindexed`. Fixed by changing `df['gender'].isin(_GENDER_ORDER)` to `sdf['gender'].isin(_GENDER_ORDER)` in the `base_rows` and `mit_rows` filters.