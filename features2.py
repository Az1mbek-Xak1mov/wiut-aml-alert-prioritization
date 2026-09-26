"""Behavioural features: recent activity vs. the customer's own baseline."""
import numpy as np
import pandas as pd

DATA = "data"
DIRS = ["kirim", "chiqim"]
TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
WINDOWS = [3, 7, 14, 30, 60]


def prep(split):
    s = pd.read_csv(f"{DATA}/{split}_signals.csv", parse_dates=["signal_sanasi"])
    t = pd.read_parquet(f"{DATA}/{split}_transactions.parquet")
    t = t.merge(s[["signal_id", "signal_sanasi"]], on="signal_id")
    v = t.tranzaksiya_vaqti
    t["lag"] = ((t.signal_sanasi - v).dt.total_seconds() / 86400).astype("float32")
    t["amt"] = np.exp(t.miqdor_indeksi).astype("float32")
    t["burst"] = ((v.dt.hour == 23) & (v.dt.minute >= 57)) | ((v.dt.hour == 0) & (v.dt.minute == 0) & (v.dt.second == 0))
    t = t.drop(columns="signal_sanasi").sort_values(["signal_id", "tranzaksiya_vaqti"]).reset_index(drop=True)
    return s, t


def window_vs_base(t, key, mask, name):
    """Stats in recent windows vs. the older baseline (lag > 60 days)."""
    sub = t[mask]
    out = {}
    base = sub[sub.lag > 60].groupby(key)
    b_n, b_mean, b_std, b_max = base.size(), base.miqdor_indeksi.mean(), base.miqdor_indeksi.std(), base.miqdor_indeksi.max()
    for w in WINDOWS:
        r = sub[sub.lag <= w].groupby(key)
        n = r.size()
        out[f"{name}_w{w}_n"] = n
        out[f"{name}_w{w}_rate"] = (n / w) / ((b_n / 120) + 0.05)
        m = r.miqdor_indeksi.mean()
        out[f"{name}_w{w}_dmean"] = m - b_mean
        out[f"{name}_w{w}_z"] = (m - b_mean) / (b_std + 0.1)
        out[f"{name}_w{w}_dmax"] = r.miqdor_indeksi.max() - b_max
        out[f"{name}_w{w}_sum"] = r.amt.sum()
    out[f"{name}_base_n"] = b_n
    out[f"{name}_base_mean"] = b_mean
    return pd.DataFrame(out)


def passthrough(t):
    """Money that comes in and quickly leaves (layering)."""
    res = {}
    t = t[["signal_id", "tranzaksiya_vaqti", "kirim_chiqim", "amt", "lag"]]
    inc = t[t.kirim_chiqim == "kirim"]
    out = t[t.kirim_chiqim == "chiqim"]
    for h in [1, 24, 72]:
        m = pd.merge_asof(
            out.sort_values("tranzaksiya_vaqti"), inc.sort_values("tranzaksiya_vaqti")[["signal_id", "tranzaksiya_vaqti", "amt"]],
            on="tranzaksiya_vaqti", by="signal_id", direction="backward",
            tolerance=pd.Timedelta(hours=h), suffixes=("", "_in"))
        m["hit"] = m.amt_in.notna()
        m["close"] = m.hit & ((m.amt / m.amt_in).between(0.8, 1.05))
        g = m.groupby("signal_id")
        res[f"pt{h}h_frac"] = g.hit.mean()
        res[f"pt{h}h_close_n"] = g.close.sum()
        res[f"pt{h}h_close_frac"] = g.close.mean()
        mr = m[m.lag <= 30].groupby("signal_id")
        res[f"pt{h}h_close_n_w30"] = mr.close.sum()
    # in/out balance over windows
    for w in [7, 30, 180]:
        x = t[t.lag <= w]
        io = x.pivot_table(index="signal_id", columns="kirim_chiqim", values="amt", aggfunc="sum")
        io = io.reindex(columns=DIRS)
        res[f"io_ratio_w{w}"] = io.chiqim / (io.kirim + 1e-3)
        res[f"io_net_w{w}"] = np.log1p(io.kirim.fillna(0)) - np.log1p(io.chiqim.fillna(0))
    return pd.DataFrame(res)


def misc(t):
    g = t.groupby("signal_id")
    res = {}
    # first appearance of each type relative to signal (new behaviour)
    for ty in TYPES:
        x = t[t.tranzaksiya_turi == ty].groupby("signal_id").lag
        res[f"{ty}_first_lag"] = x.max()
        res[f"{ty}_last_lag"] = x.min()
    # bursts
    b = t[t.burst]
    bg = b.groupby("signal_id")
    res["burst_last_lag"] = bg.lag.min()
    res["burst_first_lag"] = bg.lag.max()
    for w in [1, 7, 30]:
        res[f"burst_n_w{w}"] = b[b.lag <= w].groupby("signal_id").size()
    bday = b.assign(d=b.tranzaksiya_vaqti.dt.floor("D")).groupby(["signal_id", "d"])
    bs = bday.agg(n=("amt", "size"), s=("amt", "sum"), mx=("miqdor_indeksi", "max"),
                  nq=("tranzaksiya_turi", lambda z: (z == "naqd").sum()),
                  out=("kirim_chiqim", lambda z: (z == "chiqim").sum()))
    bsg = bs.groupby("signal_id")
    res["burst_ev_n"] = bsg.size()
    res["burst_ev_size_max"] = bsg.n.max()
    res["burst_ev_size_mean"] = bs[bs.n > 3].groupby("signal_id").n.mean()
    res["burst_ev_big_n"] = bs[bs.n > 10].groupby("signal_id").size()
    res["burst_ev_sum_max"] = bsg.s.max()
    res["burst_out_frac"] = bsg.out.sum() / bsg.n.sum()
    res["burst_naqd_frac"] = bsg.nq.sum() / bsg.n.sum()
    # amount quantile shape
    q = g.miqdor_indeksi.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    for c in q.columns:
        res[f"amt_q{int(c * 100)}"] = q[c]
    res["amt_skew"] = g.miqdor_indeksi.skew()
    res["amt_iqr"] = q[0.75] - q[0.25]
    # activity concentration (daily counts)
    d = t.assign(d=t.tranzaksiya_vaqti.dt.floor("D")).groupby(["signal_id", "d"]).size()
    dg = d.groupby("signal_id")
    res["day_n"] = dg.size()
    res["day_max"] = dg.max()
    res["day_cv"] = dg.std() / dg.mean()
    res["day_top_share"] = dg.max() / dg.sum()
    # round-ish / repeated amounts
    res["dup_amt_frac"] = t.assign(r=t.miqdor_indeksi.round(3)).groupby("signal_id").r.agg(lambda z: 1 - z.nunique() / len(z))
    # type diversity
    ct = pd.crosstab(t.signal_id, [t.kirim_chiqim, t.tranzaksiya_turi])
    ct.columns = [f"cnt_{a}_{b}" for a, b in ct.columns]
    sh = ct.div(ct.sum(axis=1), axis=0).add_prefix("sh_")
    p = sh.values
    res["type_entropy"] = pd.Series(-(p * np.log(p + 1e-12)).sum(1), index=sh.index)
    return pd.concat([pd.DataFrame(res), ct, sh], axis=1)


def build(split):
    s, t = prep(split)
    key = "signal_id"
    parts = [window_vs_base(t, key, np.ones(len(t), bool), "all")]
    for dr in DIRS:
        parts.append(window_vs_base(t, key, (t.kirim_chiqim == dr).values, dr))
    for ty in TYPES:
        parts.append(window_vs_base(t, key, (t.tranzaksiya_turi == ty).values, ty))
    for dr in DIRS:
        for ty in ["karta", "bank_otkazmasi", "naqd"]:
            parts.append(window_vs_base(t, key, ((t.kirim_chiqim == dr) & (t.tranzaksiya_turi == ty)).values, f"{dr}_{ty}"))
    parts.append(passthrough(t))
    parts.append(misc(t))
    X = s.set_index("signal_id")[[]].join(pd.concat(parts, axis=1))
    return X.astype("float32")


if __name__ == "__main__":
    for split in ["train", "test"]:
        X = build(split)
        X.to_parquet(f"{DATA}/feat2_{split}.parquet")
        print(split, X.shape)
