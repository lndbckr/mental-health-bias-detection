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

### 2026-06-13 - LLM as a Judge
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
- treatment_text extraction: from "4. Treatment Recommendations:" until "5." or end of response, whichever comes first.


