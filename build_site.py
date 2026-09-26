"""Render site/index.html from the EDA numbers and the model report."""
import json
import lightgbm as lgb
import pandas as pd
from lib import P, load, target

rep = json.load(open("submission/model_report.json"))
X = load(["feat", "feat3", "feat4"])[rep["features"]]
m = lgb.LGBMClassifier(**{**P, "n_estimators": 380}).fit(X, target(X.index))
imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
imp = (imp / imp.sum()).sort_values(ascending=False)
rep["importance"] = [[k, round(float(v), 4)] for k, v in imp.head(20).items()]
html = open("site/template.html").read()
html = html.replace("__EDA__", open("site/eda.json").read()).replace("__REPORT__", json.dumps(rep))
import os
os.makedirs("docs", exist_ok=True)
open("site/index.html", "w").write(html)
open("docs/index.html", "w").write(html)
print("site/index.html", len(html) // 1024, "KB")
