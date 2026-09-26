"""Rank all engineered features by LightGBM gain (5-fold x 3 seeds) -> data/imp_all.csv."""
from lib import cv, load, target

if __name__ == "__main__":
    d = load(["feat", "feat3", "feat4"])
    r = cv(d, target(d.index), ret_imp=True)
    print("all features: %d, OOF AUC %.4f" % (d.shape[1], r["auc"]))
    r["imp"].to_csv("data/imp_all.csv")
