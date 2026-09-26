"""Compute the numbers behind every chart on the EDA website -> site/eda.json."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

s = pd.read_csv("data/train_signals.csv", parse_dates=["signal_sanasi"])
st = pd.read_csv("data/test_signals.csv", parse_dates=["signal_sanasi"])
t = pd.read_parquet("data/train_transactions.parquet")
t = t.merge(s, on="signal_id")
v = t.tranzaksiya_vaqti
t["lag"] = (t.signal_sanasi - v).dt.total_seconds() / 86400
t["burst"] = ((v.dt.hour == 23) & (v.dt.minute >= 57)) | ((v.dt.hour == 0) & (v.dt.minute == 0) & (v.dt.second == 0))
y = s.set_index("signal_id").eskalatsiya
out = {}
r = lambda x, n=4: [round(float(a), n) for a in x]

out["overview"] = {
    "train_signals": len(s), "test_signals": len(st),
    "train_tx": len(t), "test_tx": int(pd.read_parquet("data/test_transactions.parquet", columns=["signal_id"]).shape[0]),
    "pos": int(y.sum()), "pos_rate": round(float(y.mean()), 4),
    "date_min": str(s.signal_sanasi.min().date()), "date_max": str(s.signal_sanasi.max().date()),
    "tx_min": str(v.min()), "tx_max": str(v.max()),
}

# signals and escalation rate over time
m = s.groupby(s.signal_sanasi.dt.to_period("M")).eskalatsiya.agg(["size", "mean"])
mt = st.groupby(st.signal_sanasi.dt.to_period("M")).size()
out["monthly"] = {"month": [str(p) for p in m.index], "n": m["size"].tolist(), "rate": r(m["mean"]), "n_test": mt.reindex(m.index).fillna(0).astype(int).tolist()}

# transactions per signal
n = t.groupby("signal_id").size()
edges = np.arange(0, 2400, 100)
out["tx_per_signal"] = {"edges": edges.tolist(),
                        "esc": np.histogram(n[y[y == 1].index.intersection(n.index)], edges)[0].tolist(),
                        "dis": np.histogram(n[y[y == 0].index.intersection(n.index)], edges)[0].tolist()}
nq = pd.qcut(n, 10)
out["tx_decile"] = {"label": [f"{int(i.left)}–{int(i.right)}" for i in nq.cat.categories], "rate": r(y.loc[n.index].groupby(nq).mean())}

# direction / type mix
mix = t.groupby(["tranzaksiya_turi", "kirim_chiqim"]).size().unstack()
out["mix"] = {"types": mix.index.tolist(), "kirim": mix.kirim.tolist(), "chiqim": mix.chiqim.tolist()}
sh = pd.crosstab(t.signal_id, t.tranzaksiya_turi, normalize="index")
out["type_share_by_target"] = {c: {"esc": round(float(sh.loc[y[y == 1].index, c].mean()), 4), "dis": round(float(sh.loc[y[y == 0].index, c].mean()), 4)} for c in sh.columns}

# amount distribution per type
ed = np.linspace(-3, 6, 91)
out["amount_hist"] = {"edges": r(ed, 2), **{ty: np.histogram(t.miqdor_indeksi[t.tranzaksiya_turi == ty], ed, density=True)[0].round(4).tolist() for ty in ["karta", "bank_otkazmasi", "naqd", "xalqaro"]}}

# timing: hour of day and minute within hour 23
out["hour"] = t.tranzaksiya_vaqti.dt.hour.value_counts().sort_index().tolist()
h23 = t[v.dt.hour == 23]
out["minute23"] = h23.tranzaksiya_vaqti.dt.minute.value_counts().sort_index().tolist()

# bursts
b = t[t.burst]
be = b.assign(d=b.tranzaksiya_vaqti.dt.floor("D")).groupby(["signal_id", "d"]).size()
out["burst"] = {"share_tx": round(float(t.burst.mean()), 4), "signals_with": round(float(t[t.burst].signal_id.nunique() / len(s)), 4),
                "event_size_hist": np.histogram(be, [1, 2, 5, 10, 20, 50, 100, 200, 1000])[0].tolist(),
                "event_size_edges": [1, 2, 5, 10, 20, 50, 100, 200, 1000],
                "amt_mean_burst": round(float(b.miqdor_indeksi.mean()), 3), "amt_mean_normal": round(float(t[~t.burst].miqdor_indeksi.mean()), 3)}

# activity before the signal (daily tx per signal, escalated vs dismissed)
t["lagd"] = np.floor(t.lag.clip(0, 179.999)).astype(int)
esc_ids = set(y[y == 1].index)
t["esc"] = t.signal_id.isin(esc_ids)
dl = t.groupby(["lagd", "esc"]).size().unstack() / pd.Series({False: (y == 0).sum(), True: (y == 1).sum()})
out["lag_profile"] = {"lag": dl.index.tolist(), "esc": r(dl[True], 3), "dis": r(dl[False], 3)}
dir_lag = t.groupby(["lagd", "kirim_chiqim"]).size().unstack() / len(s)
out["lag_dir"] = {"kirim": r(dir_lag.kirim, 3), "chiqim": r(dir_lag.chiqim, 3)}

# key behavioural findings
g = t.groupby(["signal_id", "kirim_chiqim", "tranzaksiya_turi"]).miqdor_indeksi.mean().unstack([1, 2])
diff = (g[("kirim", "naqd")] - t[t.tranzaksiya_turi == "bank_otkazmasi"].groupby("signal_id").miqdor_indeksi.mean()).dropna()
q = pd.qcut(diff, 10)
out["cash_vs_transfer"] = {"rate": r(y.loc[diff.index].groupby(q).mean()), "auc": round(float(roc_auc_score(y.loc[diff.index], diff)), 4)}
bo = g[("chiqim", "bank_otkazmasi")].dropna()
q = pd.qcut(bo, 10)
out["transfer_level"] = {"rate": r(y.loc[bo.index].groupby(q).mean()), "auc": round(float(1 - roc_auc_score(y.loc[bo.index], bo)), 4)}
tiny = t[(t.tranzaksiya_turi == "karta") & (t.miqdor_indeksi < -2.2)].groupby("signal_id").size().reindex(y.index).fillna(0)
tb = pd.cut(tiny, [-1, 0, 1, 2, 5, 1000], labels=["0", "1", "2", "3–5", "6+"])
out["tiny_card"] = {"label": list(tb.cat.categories), "rate": r(y.groupby(tb).mean()), "n": y.groupby(tb).size().tolist()}
fl = t[(t.tranzaksiya_turi == "bank_otkazmasi") & (t.miqdor_indeksi < -2.9)].groupby("signal_id").size().reindex(y.index).fillna(0)
fb = pd.cut(fl, [-1, 0, 1, 3, 1000], labels=["0", "1", "2–3", "4+"])
out["bank_floor"] = {"label": list(fb.cat.categories), "rate": r(y.groupby(fb).mean()), "n": y.groupby(fb).size().tolist()}

json.dump(out, open("site/eda.json", "w"))
print("ok", list(out))
