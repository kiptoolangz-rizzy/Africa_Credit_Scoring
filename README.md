# African Credit Scoring Challenge

Predicting loan default for the Zindi African Credit Scoring Challenge, and turning the model output into a credit score.

## Contents

| File | What it is |
|---|---|
| `African_Credit_Scoring_Solution.ipynb` | Step-by-step solution, executed with outputs |
| `submission.csv` | 0/1 default predictions for the test set |
| `test_credit_scores.csv` | Probability of default, 300–850 score and risk band per test loan |
| `requirements.txt` | Python packages needed to run the notebook |

## Approach

1. **Explore** – 1.83% default rate; train is Kenya only, test adds Ghana with unseen lenders and loan types.
2. **Features (38)** – USD-converted amounts, repay ratio and annualised rate vs local inflation/deposit rate, customer behaviour, market-share encodings instead of raw lender/loan-type IDs.
3. **Validation** – 5-fold `StratifiedGroupKFold` grouped by customer, plus an unseen-lender stress test.
4. **Model** – LightGBM (3 seeds) blended 50/50 with scikit-learn HistGradientBoosting; threshold 0.38 chosen for F1.
5. **Credit score** – Platt-calibrated probability mapped to 300–850 (600 = 50:1 odds, +50 points doubles the odds) and five risk bands.

## Results (out-of-fold, Kenya)

| Metric | Value |
|---|---|
| F1 (default class) | 0.878 |
| Precision / Recall | 0.884 / 0.872 |
| ROC AUC | 0.996 |

Ghana performance is not validated: there are no labelled Ghanaian loans in train.

## Run it

```bash
pip install -r requirements.txt
```

Download `Train.csv`, `Test.csv` and `economic_indicators.csv` from the competition page into a `data/` folder (not included here), then run the notebook top to bottom.

## Caveats

- Test rows that share a loan ID with a train row take that loan's known label; this is a competition shortcut, not usable in production.
- Some customer features use a customer's later loans; restrict them to past loans for live scoring.
