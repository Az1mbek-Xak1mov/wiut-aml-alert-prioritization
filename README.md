# AML Alert Prioritization: Team 10x Engineers (EC6D41A5)

WIUT Hackathon 2026, Fintech / AI in Finance track. The model predicts the probability that an AML alert gets escalated (the metric is ROC-AUC). The full task description is in [TASK.md](TASK.md).

**Result:** cross-validated ROC-AUC **0.663** (stratified 5-fold × 5 seeds).

## Deliverables
| | |
|---|---|
| Predictions | `submission/team_EC6D41A5.csv` |
| EDA website | https://az1mbek-xak1mov.github.io/wiut-aml-alert-prioritization/ (source: `site/template.html`, built into `docs/index.html`) |
| Reproducible notebook | `solution.ipynb` |

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt jupyter
# put the organisers' files into data/ :
#   train_signals.csv test_signals.csv train_transactions.parquet test_transactions.parquet sample_submission.csv
.venv/bin/jupyter nbconvert --to notebook --execute solution.ipynb   # ~25 min on 12 cores
```

## Pipeline
| File | Purpose |
|---|---|
| `features.py` | Counts and amount stats per direction, type and direction×type; time windows; bursts; gaps between transactions |
| `features3.py` | **Relative** features: each group compared with the customer's own overall level, plus pairwise group differences |
| `features4.py` | Amount histograms per type × burst/normal; tiny-card and floor counts |
| `features2.py` | Recent-vs-baseline activity and pass-through features. Tested but not used (weaker) |
| `select_features.py` | Ranks the 857 features by LightGBM gain; the model keeps the top 100 |
| `train.py` | LightGBM + CatBoost + logistic regression, 5-fold × 5 seeds, blended 0.8/0.1/0.1 → submission |
| `eda_data.py`, `build_site.py`, `site/template.html` | EDA numbers and the static website |

## Key findings
1. The signal is weak but real. 17.2% of alerts are escalated, with no trend over time, and no single feature goes above AUC 0.62.
2. **Relative amount size matters, absolute size doesn't.** Cash deposits and card spend that are large compared with the customer's own bank transfers raise the escalation rate from 10% to 29%.
3. **Tiny card transactions** (amount index below −2.2, typical of card testing) raise the escalation rate from 16% to about 30%.
4. **Bursts before midnight** appear in 99% of alerts and show how alerts get triggered, but timing and recency features barely separate escalated from dismissed alerts.
