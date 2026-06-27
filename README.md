# Gender & Age Bias in LLM-Based Depression Diagnosis

**Course:** Gender, Diversity & AI — University of Klagenfurt, SS2026

> This codebase was generated via the [Promptotyping](https://github.com/DigitalHumanitiesCraft/promptotyping-skill) methodology.

---

## Requirements

- Python 3.13, Ollama running locally with `llama3:8b` and `mistral:7b` pulled
- `pip install requests>=2.31 pandas>=2.2 matplotlib>=3.8`

---

## How to Run

**Prototype (1 run per variant):**
```bash
python main.py --prototype
```

**One batch** (390 LLM calls + 390 judge calls):
```bash
python main.py --batch --severity mild --models mistral:7b --judge-model llama3:8b
```
Repeat for all 8 partitions (4 severities × 2 models), then analyse once:
```bash
python main.py --step analyse
```

**Full experiment in one go:**
```bash
python main.py
```

Interrupted batches can be resumed — already-written raw responses are skipped automatically.

---

## Experiment Scale

| Condition | Calls |
|---|---|
| base + mitigation (5 gender × 3 age × 4 severity × 2 type × 2 model × 10 runs) | 2,400 |
| neutral_full / neutral_age / neutral_gender | 720 |
| **Total** | **3,120** |

Plus ~3,120 judge calls.

---

## Outputs

| Path | Description |
|---|---|
| `data/raw_responses/{variant_id}.json` | One file per run |
| `data/processed/results_{sev}_{model_slug}.csv` | Parsed + scored partition per batch |
| `results/aggregated_results.csv` | Descriptive stats per variant |
| `results/comparisons_{gender,age,intersectionality}.csv` | Observed differences between conditions |
| `results/plots/*.png` | Visualizations |
