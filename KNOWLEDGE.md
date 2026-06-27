# knowledge.md – Mental Health Bias in LLMs

**Project:** Gender & Age Bias in LLM-Based Depression Diagnosis  
**Course:** Gender, Diversity & AI – University of Klagenfurt, SS2026  

---

## Research Question

> Do LLMs produce biased outputs when responding to mental health concerns, and does the bias differ along gender and age dimensions?

Specifically: Do LLMs show gender and age bias when diagnosing depression from identical symptom presentations?

---

## Domain Knowledge

### Why Mental Health?
Unlike specialized AI tools used by professionals, general-purpose chatbots (ChatGPT, Claude, Gemini) are directly accessible by anyone seeking health information. Many people use these chatbots as informal "digital doctors" or "digital therapists." Bias in their outputs has direct public health implications. Research on gender bias in LLM-mediated mental health contexts remains limited.

### Key Clinical Context
- Depression diagnosis should be based purely on reported symptoms, not patient demographics.
- Severity and impairment scores should be identical across gender/age groups for identical symptom presentations.
- Standard clinical tools: DSM-5 criteria for MDD; PHQ-9 for severity screening.

### LLM Bias Types (relevant to this project)
- **Stereotyping** – group-based generalizations applied to diagnosis/treatment.
- **Treatment Bias** – recommending different treatment intensities for identical symptoms based on demographics.
- **Minimizing / Dismissive Language** – downplaying symptoms for certain groups (e.g., "just stress," "normal for her age").

### Known LLM Risks in this Domain
- Provider **blocking**: Gemini API has been documented to block sensitive mental health prompts at default safety settings.
- **Sycophancy trap**: Never ask "is this biased?" in evaluation prompts. Always evaluate against a rubric with observable indicators only.
- **Diversity Instruction Paradox**: Diversity-aware prompts can produce *higher* bias scores than neutral ones (FAIR-SW-Bench finding). 
- LLM outputs are **stochastic** – run each variant N ≥ 10 times to capture variance.

### Statistical Approach: Descriptive, Not Inferential
The N ≥ 10 runs per variant are repeated stochastic generations from a fixed model — they characterise the model's own output distribution, not a random sample drawn from a population. With no population to generalise to, and every condition of interest enumerated exhaustively, inferential tests (p-values, ANOVA, t-tests, chi-square) do not apply. The project reports **descriptive** statistics: means, SDs, ranges, frequencies, proportions, and the observed differences between conditions, with spread shown directly. See requirements.md § Analysis.

---

## Expected Findings

This experiment may reveal that llama3 and mistral:
- detects depression more often in women
- under-detects depression in men
- normalizes depression in elderly adults (65+)
- over-medicalizes women
- responds differently to trans and non-binary variants despite identical symptom text

---

## Literature

### Chansiri, Wei & Chor (2024)
*Addressing Gender Bias: A Fundamental Approach to AI in Mental Health.* IBDAP, IEEE.  
- Tested GPT-3.5 and GPT-4 on gender bias in BPD/NPD using association and text-generation tasks.  
- Both models showed significant gender bias, particularly against women, exceeding clinical evidence.  
- Relevance: validates gender bias signal in diagnostic LLM outputs; methodology inspiration for text-generation approach.

### Schnepper et al. (2025)
*Exploring Biases of LLMs in Eating Disorder Case Vignettes.* JMIR Mental Health 12.  
- ChatGPT-4 showed gender bias in mental health quality-of-life scores for identical ED presentations; no significant sexual orientation bias.  
- MentaLLaMA produced unreliable results (too many failures to score).  
- Relevance: direct methodological model – modified real-case vignettes with demographic variants; used standardized assessment rubrics.

### Lawrence et al. (2024)
*Opportunities and Risks of LLMs in Mental Health.* JMIR Mental Health 11.  
- Identified LLM risks including perpetuating inequalities/stigma, insufficient reliability, lack of transparency.  
- Domain-specific models (MentalBERT) outperform general-purpose LLMs in classification tasks.  
- Relevance: frames ethical stakes; informs why general-purpose LLM bias matters more for public health than specialist tools.

