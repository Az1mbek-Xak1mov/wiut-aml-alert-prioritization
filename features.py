"""Aggregate each signal's transaction history into one feature row."""
import numpy as np
import pandas as pd

DATA = "data"


def load(split):
    s = pd.read_csv(f"{DATA}/{split}_signals.csv", parse_dates=["signal_sanasi"])
    t = pd.read_parquet(f"{DATA}/{split}_transactions.parquet")
    return s, t


def _agg(df, prefix):
    g = df.groupby("signal_id").miqdor_indeksi
    out = pd.DataFrame({
        f"{prefix}_n": g.size(),
        f"{prefix}_mean": g.mean(),
        f"{prefix}_std": g.std(),
        f"{prefix}_max": g.max(),
        f"{prefix}_min": g.min(),
        f"{prefix}_med": g.median(),
        f"{prefix}_q90": g.quantile(0.9),
        f"{prefix}_sumexp": df.assign(e=np.exp(df.miqdor_indeksi)).groupby("signal_id").e.sum(),
    })
    return out


def build(split):
    s, t = load(split)
    t = t.merge(s[["signal_id", "signal_sanasi"]], on="signal_id")
    v = t.tranzaksiya_vaqti
    t["lag"] = (t.signal_sanasi - v).dt.total_seconds() / 86400
    t["burst"] = (v.dt.hour == 23) & (v.dt.minute >= 57) | ((v.dt.hour == 0) & (v.dt.minute == 0) & (v.dt.second == 0))
    t["day"] = v.dt.floor("D")
    t["hour"] = v.dt.hour

    parts = [_agg(t, "all")]
    for d in ["kirim", "chiqim"]:
        parts.append(_agg(t[t.kirim_chiqim == d], d))
    for ty in ["karta", "bank_otkazmasi", "naqd", "xalqaro"]:
        parts.append(_agg(t[t.tranzaksiya_turi == ty], ty))
    for d in ["kirim", "chiqim"]:
        for ty in ["karta", "bank_otkazmasi", "naqd", "xalqaro"]:
            sub = t[(t.kirim_chiqim == d) & (t.tranzaksiya_turi == ty)]
            parts.append(sub.groupby("signal_id").miqdor_indeksi.agg(["size", "mean"]).add_prefix(f"{d}_{ty}_"))
    for w in [1, 7, 14, 30, 60, 90]:
        parts.append(_agg(t[t.lag <= w], f"w{w}"))
        for d in ["kirim", "chiqim"]:
            parts.append(t[(t.lag <= w) & (t.kirim_chiqim == d)].groupby("signal_id").size().rename(f"w{w}_{d}_n"))
    parts.append(_agg(t[t.burst], "burst"))
    parts.append(_agg(t[~t.burst], "nonburst"))

    # daily activity profile
    daily = t.groupby(["signal_id", "day"]).size()
    dg = daily.groupby("signal_id")
    parts.append(pd.DataFrame({"days_active": dg.size(), "daily_max": dg.max(), "daily_mean": dg.mean(), "daily_std": dg.std()}))
    bd = t[t.burst].groupby(["signal_id", "day"]).size().groupby("signal_id")
    parts.append(pd.DataFrame({"burst_days": bd.size(), "burst_daymax": bd.max()}))

    # time span / gaps
    tg = t.sort_values(["signal_id", "tranzaksiya_vaqti"]).groupby("signal_id")
    parts.append(pd.DataFrame({
        "lag_min": tg.lag.min(), "lag_max": tg.lag.max(), "lag_mean": tg.lag.mean(),
        "night_frac": t.assign(x=t.hour.between(0, 5)).groupby("signal_id").x.mean(),
    }))
    gaps = t.sort_values(["signal_id", "tranzaksiya_vaqti"]).groupby("signal_id").tranzaksiya_vaqti.diff().dt.total_seconds()
    t["gap"] = gaps
    gg = t.groupby("signal_id").gap
    parts.append(pd.DataFrame({"gap_med": gg.median(), "gap_min": gg.min(), "gap_max": gg.max(),
                               "gap_lt60_frac": t.assign(x=t.gap < 60).groupby("signal_id").x.mean()}))

    X = s.set_index("signal_id")[["signal_sanasi"]].join(pd.concat(parts, axis=1))
    X["month"] = X.signal_sanasi.dt.month
    X["dow"] = X.signal_sanasi.dt.dayofweek
    X["doy"] = X.signal_sanasi.dt.dayofyear
    X["t_ord"] = (X.signal_sanasi - pd.Timestamp("2025-01-01")).dt.days
    X = X.drop(columns="signal_sanasi")
    # ratios
    X["chiqim_kirim_ratio"] = X.chiqim_n / X.kirim_n
    X["out_in_amt_ratio"] = X.chiqim_sumexp / X.kirim_sumexp
    for w in [1, 7, 14, 30]:
        X[f"w{w}_share"] = X[f"w{w}_n"] / X.all_n
        X[f"w{w}_rate_vs_base"] = (X[f"w{w}_n"] / w) / (X.all_n / 180)
    X["naqd_share"] = X.naqd_n / X.all_n
    X["xalqaro_share"] = X.xalqaro_n / X.all_n
    X["burst_share"] = X.burst_n / X.all_n
    y = s.set_index("signal_id").get("eskalatsiya")
    return X, y


if __name__ == "__main__":
    for split in ["train", "test"]:
        X, y = build(split)
        if y is not None:
            X["eskalatsiya"] = y
        X.to_parquet(f"{DATA}/feat_{split}.parquet")
        print(split, X.shape)
