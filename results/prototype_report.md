# Prototype Results Report
**Gender & Age Bias in LLM-Based Depression Diagnosis**  
**Scope: Mild severity · llama3:8b · N = 390 responses** 

> **Data note:** Bar charts and heatmap plots use **base + mitigation combined** (N = 60 per gender group, 20 per prompt_type × gender cell).
> Box plots and intersectionality plots use **base condition only** (N = 30 per gender group).
> All numbers in this report match the corresponding plots exactly.

---

## Overview

The prototype ran 390 calls across 39 experimental conditions (5 genders × 3 ages × base + mitigation + 3 neutral prompt types, 10 runs each). Every field was successfully extracted — **0% missing data** across all six scored outcomes. Two statistically significant effects were found after Bonferroni correction.

---

## Q1 — Depression Diagnosis Rate

**100% detection across every condition.** Every response diagnosed depression regardless of gender, age, or prompt type. This is a ceiling effect: mild-severity vignettes are symptomatically unambiguous enough that llama3:8b diagnoses depression without exception. There is no measurable bias in *whether* depression is detected at this severity level. Q1 becomes informative when moderate, severe, and ambiguous severities are added in the full experiment.

---

## Q2 — Symptom Severity Estimate (0–10)
*Basis: base + mitigation combined*

| Gender | Base | Mitigation |
|---|---|---|
| woman | 5.67 | 5.87 |
| man | 5.80 | 5.93 |
| non-binary | 5.73 | 5.80 |
| trans-woman | 5.70 | **6.00** |
| trans-man | 5.93 | **6.00** |

**Mean by gender (base + mitigation):** man = 5.87, non-binary = 5.77, trans-man = 5.97, trans-woman = 5.85, woman = 5.77

Scores cluster tightly around 5.7–6.0. Trans variants received the highest severity ratings under the mitigation condition (both reached 6.0 with zero variance). The Welch ANOVA showed a raw effect (F = 3.06, p = 0.019) but this **did not survive Bonferroni correction** (p_corrected = 0.23).

**By age group (base + mitigation):** 18–25 = 5.73, 40–55 = 5.93, 65+ = 5.87. No significant age effect after correction.

**Interpretation:** No statistically robust severity bias at prototype scale. The trans-variant elevation is a signal worth watching in the full experiment.

---

## Q3 — Functional Impairment Estimate (0–10) ★ Significant
*Basis: base + mitigation combined*

### By age group — Welch ANOVA: F = 10.26, p = 0.0001, p_corrected = 0.0012 ✅

| Age group | Mean impairment (base + mitigation) |
|---|---|
| 18–25 (young adults) | **4.84** |
| 40–55 (middle-aged) | 5.33 |
| 65+ (elderly) | **5.49** |

The model rates elderly patients as significantly more functionally impaired than young adults — despite identical symptom text. The same mild depression symptoms are judged as more disabling when the patient is 70 years old than when they are 22. This is a clear **age bias** signal.

### By gender (base + mitigation)

| Gender | Mean impairment |
|---|---|
| man | **5.00** — lowest |
| trans-man | 5.20 |
| woman | 5.23 |
| trans-woman | 5.30 |
| non-binary | 5.37 |

Men receive the lowest impairment ratings for identical symptoms. This gender effect **did not survive Bonferroni correction** (p_corrected = 1.0) and requires the full experiment to confirm.

### By gender — base condition only (box plot basis)

| Gender | Base mean | Base SD |
|---|---|---|
| man | **4.77** | 1.07 |
| woman | 5.10 | 1.24 |
| trans-woman | 5.27 | 1.17 |
| trans-man | 5.27 | 1.23 |
| non-binary | 5.27 | 1.20 |

---

## Q4a — Treatment Intensity Score (0–10)
*Basis: base + mitigation combined*

| Gender | Base | Mitigation | Combined mean |
|---|---|---|---|
| man | 7.24 | 7.03 | 7.14 |
| non-binary | 7.07 | 7.24 | 7.15 |
| trans-man | 7.30 | 7.00 | 7.15 |
| trans-woman | 7.29 | 7.23 | 7.26 |
| woman | 7.18 | 7.03 | 7.10 |

**By age group:** 18–25 = 7.28, 40–55 = 7.14, 65+ = 7.06. Slight downward trend for older patients (treated less intensively) but not statistically significant.

Treatment scores are consistently high and narrow across all groups. The typical recommendation set (therapy + SSRI + lifestyle) produces a weighted score around 7.1–7.3. No significant demographic differences were found.

---

## Q4b — Treatment Type Profile
*Basis: base + mitigation combined — matches heatmap plot*

| Gender | Clinical / therapy | Pharmacological / medication |
|---|---|---|
| man | 55.0% | **45.0%** |
| non-binary | 65.0% | 35.0% |
| trans-man | 63.3% | 36.7% |
| trans-woman | 70.0% | 30.0% |
| **woman** | **73.3%** | 26.7% |

**This is a consistent directional pattern.** Women receive therapy recommendations in 73% of cases, while men are directed toward medication in 45% of cases — nearly twice the rate. The gender gradient runs woman → trans-woman → trans-man → non-binary → man with decreasing clinical recommendation rates. This mirrors real-world clinical gender stereotypes where women are steered toward talk therapy and men toward pharmacotherapy.

---

## Q5 — Minimizing / Dismissive Language (0–10) ★ Significant
*Basis: base + mitigation combined*

### By prompt type — Welch ANOVA: F = 10.33, p = 0.0015, p_corrected = 0.018 ✅

| Condition | Mean minimizing score |
|---|---|
| Base (no constraint) | 3.52 |
| Mitigation (constraint added) | **2.63** |

The mitigation prompt ("Assess only the reported symptoms. Demographic information is irrelevant.") **significantly reduces dismissive language** by approximately 0.9 points. The constraint is doing measurable work.

### By gender — base + mitigation combined

| Gender | Combined mean |
|---|---|
| man | **2.42** — lowest |
| non-binary | 3.00 |
| woman | 3.03 |
| trans-woman | 3.42 |
| trans-man | **3.52** — highest |

### By gender — base vs mitigation breakdown (box plot basis: base only)

| Gender | Base | Mitigation | Reduction |
|---|---|---|---|
| trans-man | **4.47** | 2.57 | −1.90 |
| trans-woman | **4.10** | 2.73 | −1.37 |
| woman | 3.40 | 2.67 | −0.73 |
| man | 2.80 | 2.03 | −0.77 |
| non-binary | 2.83 | 3.17 | +0.34 |

Trans variants receive the most dismissive language in the base condition and benefit most from the mitigation constraint. Non-binary is the only group where the constraint slightly increases minimizing language (small effect, likely noise at N=30).

---

## Q6 — Person Framing Label
*Basis: base + mitigation combined — matches stacked bar plot*

| Gender | Neutral | Distanced | Medicalized | Gendered |
|---|---|---|---|---|
| non-binary | **56.7%** | 28.3% | 1.7% | 13.3% |
| woman | 13.3% | 20.0% | 13.3% | **53.3%** |
| man | 15.0% | **51.7%** | 13.3% | 20.0% |
| trans-woman | 15.0% | 31.7% | **25.0%** | 28.3% |
| trans-man | 21.7% | 41.7% | **21.7%** | 15.0% |

Three distinct patterns emerge:

- **Women** are predominantly referred to as "the woman" (53.3% gendered) — gender is foregrounded in clinical reasoning where it is clinically irrelevant.
- **Men** are described with distancing language — "the individual" (51.7% distanced), suggesting clinical detachment when the patient is male.
- **Trans individuals** show elevated medicalization (21–25%) — the model shifts to a clinical-role framing ("the patient", "the client") when gender identity is made explicit.
- **Non-binary** individuals receive the most neutral language (56.7%) — the model most often uses "the person" or they/them without identity-foregrounding.

---

## Neutral Baseline Comparisons
*All values: mild severity only*

Comparing demographic conditions against the symptom-only baseline (neutral_full) reveals the direction and magnitude of demographic cue effects:

| Metric | Base | Neutral-full (no demo) | Neutral-age (gender only) | Neutral-gender (age only) |
|---|---|---|---|---|
| Severity | 5.77 | 5.60 | 5.80 | 5.93 |
| Impairment | 5.13 | 5.30 | 5.04 | 5.17 |
| Treatment | 7.22 | **6.65** | 7.22 | 7.31 |
| Minimizing | 3.52 | 3.80 | 3.40 | **3.23** |

Key observations:
- **Treatment score drops by −0.57** when all demographics are removed (neutral_full vs base), suggesting demographic cues inflate treatment intensity recommendations.
- **Removing gender alone** (neutral_gender) reduces minimizing language by −0.29 — the gender cue specifically triggers more dismissive language.
- Removing age alone (neutral_age) has minimal impact on most outcomes, consistent with the impairment finding that age affects perceived impairment but not other outcomes.

---

## Intersectionality — Gender × Age Interaction (base condition only)

### Impairment score by gender × age

| Gender | 18–25 | 40–55 | 65+ |
|---|---|---|---|
| man | 4.4 | 4.6 | **5.3** |
| non-binary | 4.7 | 5.3 | 5.8 |
| trans-man | 4.6 | **5.8** | 5.4 |
| trans-woman | 5.4 | 4.8 | 5.6 |
| woman | 4.5 | 5.3 | 5.5 |

### Minimizing score by gender × age

| Gender | 18–25 | 40–55 | 65+ |
|---|---|---|---|
| man | 4.2 | 2.2 | 2.0 |
| non-binary | 3.4 | 2.6 | 2.5 |
| trans-man | 4.0 | 3.8 | **5.6** |
| trans-woman | 3.1 | 4.0 | **5.2** |
| woman | 2.8 | 2.8 | 4.6 |

A notable intersectional pattern: **elderly trans individuals (65+) receive substantially more dismissive language** — trans-man at 65+ scores 5.6 and trans-woman at 65+ scores 5.2, well above any other gender × age combination. The combination of trans identity and old age appears to compound the minimizing effect beyond what either factor produces alone.

---

## Summary of Key Findings

| Finding | Direction | Statistical status |
|---|---|---|
| Impairment rated higher for elderly | 65+ (5.49) > 18–25 (4.84), Δ = +0.65 pts | ✅ Significant (p_corrected = 0.001) |
| Mitigation reduces dismissive language | Base (3.52) > Mitigation (2.63), Δ = −0.89 pts | ✅ Significant (p_corrected = 0.018) |
| Trans variants receive most dismissive language | trans-man: 3.52, man: 2.42 (combined) | Trend — not yet significant |
| Women steered toward therapy; men toward medication | 73.3% vs 55.0% clinical recommendation | Descriptive — χ² needed |
| Women framed by gender identity; men distanced; trans medicalized | Q6 pattern consistent across 60 runs per group | Descriptive |
| Men rated lowest impairment for identical symptoms | man: 5.00 vs non-binary: 5.37 | Trend — not yet significant |
| Elderly trans individuals: highest minimizing language | trans-man 65+: 5.6; trans-woman 65+: 5.2 | Intersectional signal |
| Demographic cues inflate treatment recommendations | neutral_full −0.57 vs base | Descriptive |

---

## Limitations & Next Steps

- **Prototype only:** mild severity and llama3:8b only. Most gender effects are trends that need larger samples and more severity levels to reach corrected significance.
- **100% diagnosis rate** at mild severity removes Q1 as a discriminating variable — moderate, severe, and ambiguous conditions will likely show differential detection rates.
- **No cross-model comparison yet** — adding mistral:7b will reveal whether these patterns are llama3-specific artefacts or general LLM behaviour.
- **Self-judging limitation:** llama3:8b judged its own outputs for Q4a/Q4b in the prototype. Full experiment uses true cross-model judging.

### Commands to extend the experiment

```bash
# Add mistral:7b to prototype (cross-model check, ~390 new calls)
python main.py --prototype --models mistral:7b --step run
python main.py --step parse
python main.py --step analyse

# Run the full experiment (all 4 severities, both models, ~2730 additional calls)
python main.py --step run
python main.py --step parse
python main.py --step analyse
```
