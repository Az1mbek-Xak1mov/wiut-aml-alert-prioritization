# WIUT Hackathon 2026: Elimination Task

**Track:** Fintech / AI in Finance
**Task:** AML Alert Prioritization
**Team:** 10x Engineers (`EC6D41A5`)
**Deadline:** Sunday, 27 September 2026, 23:59 (Tashkent time). Late submissions are not assessed.
**Terms accepted:** 23 Sep 2026, 11:44 (version 1.0)

> No more than 5 teams from the same university can be selected for the final.

Source: https://hackathon.wiut.uz/team/task/

---

## Problem

A financial-sector monitoring unit in Uzbekistan receives automated **AML (anti-money-laundering) alerts** generated from customers' historical transaction activity. Specialists review each alert and either **dismiss** it or **escalate** it for further investigation.

**Goal:** build an ML model that estimates the **probability that each alert in the hidden test set will be escalated.**

The data is synthetic, transformed, and localized. It contains no real customer or bank records.

## Data files

| File | Description |
|---|---|
| `train_signals.csv` | One row per training alert |
| `train_transactions.parquet` | Historical transactions linked to training alerts (many rows per alert) |
| `test_signals.csv` | One row per hidden test alert (no target) |
| `test_transactions.parquet` | Historical transactions linked to test alerts |
| `sample_submission.csv` | Example submission format |

### `train_signals.csv` / `test_signals.csv`

| Column | Meaning |
|---|---|
| `signal_id` | Unique alert ID |
| `signal_sanasi` | Alert date (*sana* = date) |
| `eskalatsiya` | **Target**: 1 = escalated, 0 = dismissed (train only) |

### `*_transactions.parquet`

| Column | Meaning |
|---|---|
| `signal_id` | Alert ID (foreign key) |
| `tranzaksiya_vaqti` | Transaction timestamp |
| `kirim_chiqim` | Direction: `kirim` (incoming) / `chiqim` (outgoing) |
| `tranzaksiya_turi` | Type: `karta` (card), `bank_otkazmasi` (bank transfer), `naqd` (cash), `xalqaro` (international) |
| `miqdor_indeksi` | Standardized transaction-size indicator |

The transaction table is **relational**, so each `signal_id` can have many transactions. Teams decide how to aggregate or model this history (feature engineering).

## Evaluation

- Metric: **ROC-AUC** (higher is better).
- Predictions must be real-valued probabilities in `[0, 1]`, not hard labels.
- The highest valid ROC-AUC scores move on to the next stage, provided all deliverables are complete and the work passes reproducibility and eligibility checks.

## Deliverables (all 3 required)

1. **`team_EC6D41A5.csv`**: predictions for the test set.
2. **A public EDA website URL**: must be reachable without logging in.
3. **A reproducible Jupyter notebook.**

Only **one** official prediction submission is accepted per team.

### Submission format

```csv
signal_id,ehtimollik
SG_000001,0.1842
SG_000002,0.8271
SG_000003,0.0537
```

Rules:
- Exactly two columns, in this order: `signal_id,ehtimollik` (*ehtimollik* = probability)
- Exactly one row per test `signal_id`: no duplicates, no missing IDs, no unknown IDs
- No missing predictions; `0 <= ehtimollik <= 1`
- CSV only, with no index column
- Filename: `team_<TEAM_ID>.csv` → `team_EC6D41A5.csv`

## EDA website requirements

You can build it with any tech (Streamlit, Gradio, GitHub Pages, Vercel, Netlify, …). At minimum it must include:

- A short description of the approach
- An overview of the dataset and its structure
- Several meaningful EDA visualizations
- Key observations and insights from the transaction history
- Target distribution analysis and relevant behavioral patterns
- The features or modeling ideas that came out of the EDA
- A brief conclusion with the most important findings

Suggested analyses:
- Transaction activity over time
- Incoming vs. outgoing behavior
- Differences between transaction types
- Transaction-size distributions
- Activity immediately before a signal
- Behavioral differences between escalated and dismissed signals

## Rules

- Any legitimate ML approach and local feature engineering is allowed.
- **Prohibited:** using hidden labels, external copies of the target labels, or trying to get organizer-only data.
- The EDA website does not replace the prediction CSV. Both are required.
