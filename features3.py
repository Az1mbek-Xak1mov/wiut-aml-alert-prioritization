"""Relative amount features: each direction/type group compared with the customer's own level."""
import itertools

import numpy as np
import pandas as pd

DATA = "data"
GROUPS = [f"{d}_{t}" for d in ["kirim", "chiqim"] for t in ["karta", "bank_otkazmasi", "naqd", "xalqaro"]]
STATS = ["mean", "median", "q25", "q75", "q90", "max", "std"]


def stats(t, key):
    g = t.groupby(key).miqdor_indeksi
    return pd.DataFrame({
        "n": g.size(), "mean": g.mean(), "median": g.median(), "q25": g.quantile(0.25),
        "q75": g.quantile(0.75), "q90": g.quantile(0.9), "max": g.max(), "std": g.std(),
    })


def build(split):
    s = pd.read_csv(f"{DATA}/{split}_signals.csv", parse_dates=["signal_sanasi"])
    t = pd.read_parquet(f"{DATA}/{split}_transactions.parquet", columns=["signal_id", "kirim_chiqim", "tranzaksiya_turi", "miqdor_indeksi", "tranzaksiya_vaqti"])
    t = t.merge(s[["signal_id", "signal_sanasi"]], on="signal_id")
    t["lag"] = (t.signal_sanasi - t.tranzaksiya_vaqti).dt.total_seconds() / 86400
    t["grp"] = t.kirim_chiqim + "_" + t.tranzaksiya_turi
    t["dt"] = t.kirim_chiqim + "_" + t.tranzaksiya_turi.where(t.tranzaksiya_turi != "xalqaro", "xalqaro")

    parts = []
    overall = stats(t, "signal_id")
    per = stats(t, ["signal_id", "grp"]).unstack("grp")
    per.columns = [f"{g}_{st}" for st, g in per.columns]
    parts.append(per)
    for ty in ["karta", "bank_otkazmasi", "naqd", "xalqaro"]:
        x = stats(t[t.tranzaksiya_turi == ty], "signal_id").add_prefix(f"{ty}_")
        parts.append(x)
    for dr in ["kirim", "chiqim"]:
        parts.append(stats(t[t.kirim_chiqim == dr], "signal_id").add_prefix(f"{dr}_"))
    X = pd.concat([overall.add_prefix("all_")] + parts, axis=1)

    # relative to the customer's overall level
    rel = {}
    for g in GROUPS + ["karta", "bank_otkazmasi", "naqd", "xalqaro", "kirim", "chiqim"]:
        for st in ["mean", "median", "q90", "max"]:
            rel[f"rel_{g}_{st}"] = X[f"{g}_{st}"] - X[f"all_{st}"]
        rel[f"relz_{g}_mean"] = (X[f"{g}_mean"] - X["all_mean"]) / X["all_std"]
        rel[f"share_{g}"] = X[f"{g}_n"] / X["all_n"]
    # pairwise differences of group means / medians
    for a, b in itertools.combinations(GROUPS + ["karta", "bank_otkazmasi", "naqd"], 2):
        rel[f"d_{a}__{b}"] = X[f"{a}_mean"] - X[f"{b}_mean"]
        rel[f"dm_{a}__{b}"] = X[f"{a}_median"] - X[f"{b}_median"]
    # same in the recent 30 days
    r = t[t.lag <= 30]
    rp = r.groupby(["signal_id", "grp"]).miqdor_indeksi.mean().unstack()
    ra = r.groupby("signal_id").miqdor_indeksi.mean()
    for g in rp.columns:
        rel[f"rel30_{g}_mean"] = rp[g] - ra
    X = pd.concat([X, pd.DataFrame(rel)], axis=1)
    X = s.set_index("signal_id")[[]].join(X)
    return X.astype("float32")


if __name__ == "__main__":
    for split in ["train", "test"]:
        X = build(split)
        X.to_parquet(f"{DATA}/feat3_{split}.parquet")
        print(split, X.shape)
