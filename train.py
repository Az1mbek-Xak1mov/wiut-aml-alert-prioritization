"""Final model: LightGBM + CatBoost + logistic regression ensemble on the top-100 features.

Writes out-of-fold predictions, the blend weights and submission/team_EC6D41A5.csv.
"""
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

import lightgbm as lgb
from lib import P, StratifiedKFold, load, roc_auc_score, target

TEAM_ID = "EC6D41A5"
SETS = ["feat", "feat3", "feat4"]
N_FEATURES = 100
SEEDS = [0, 1, 2, 3, 4]

imp = pd.read_csv("data/imp_all.csv", index_col=0).iloc[:, 0]
cols = list(imp.index[:N_FEATURES])
X = load(SETS)[cols]
Xt = load(SETS, "test").reindex(columns=cols)
y = target(X.index)


def lgbm():
    return lgb.LGBMClassifier(**{**P, "n_estimators": 380})


def catb(seed):
    return CatBoostClassifier(iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=10, rsm=0.5,
                              verbose=0, random_seed=seed, thread_count=12)


def logreg():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0),
                         LogisticRegression(C=0.003, max_iter=3000))


MODELS = {"lgbm": lambda s: lgbm().set_params(random_state=s), "cat": catb, "lr": lambda s: logreg()}

oof = {k: np.zeros(len(X)) for k in MODELS}
pred = {k: np.zeros(len(Xt)) for k in MODELS}
for seed in SEEDS:
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        for name, make in MODELS.items():
            m = make(seed)
            m.fit(X.iloc[tr], y.iloc[tr])
            oof[name][va] += m.predict_proba(X.iloc[va])[:, 1] / len(SEEDS)
            pred[name] += m.predict_proba(Xt)[:, 1] / (5 * len(SEEDS))
    print("seed", seed, {k: round(roc_auc_score(y, v), 4) for k, v in oof.items()}, flush=True)

scores = {k: roc_auc_score(y, v) for k, v in oof.items()}
print("OOF AUC per model:", scores)

# weighted probability blend: LightGBM is strongest; small weights on the others add robustness
# at equal OOF AUC, and averaging probabilities keeps the output a calibrated-looking probability
w = {"lgbm": 0.8, "cat": 0.1, "lr": 0.1}
best = (roc_auc_score(y, sum(w[k] * oof[k] for k in oof)), w)
print("blend OOF AUC %.4f weights %s" % best)
final = np.clip(sum(w[k] * pred[k] for k in pred), 0, 1)

os.makedirs("submission", exist_ok=True)
sub = pd.read_csv("data/test_signals.csv")[["signal_id"]]
sub["ehtimollik"] = pd.Series(final, index=Xt.index).loc[sub.signal_id].values.round(6)
sub.to_csv(f"submission/team_{TEAM_ID}.csv", index=False)
pd.DataFrame(oof, index=X.index).assign(y=y).to_parquet("data/oof.parquet")
pd.DataFrame(pred, index=Xt.index).to_parquet("data/test_pred.parquet")
json.dump({"oof_auc": scores, "blend_auc": best[0], "weights": w, "features": cols},
          open("submission/model_report.json", "w"), indent=2, default=float)
print("wrote", len(sub), "rows")
