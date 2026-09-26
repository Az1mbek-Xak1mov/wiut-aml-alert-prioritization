"""Amount histograms per direction/type, split into burst vs. normal activity."""
import numpy as np
import pandas as pd

DATA = "data"
EDGES = np.array([-np.inf, -2.2, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf])


def build(split):
    s = pd.read_csv(f"{DATA}/{split}_signals.csv", parse_dates=["signal_sanasi"])
    t = pd.read_parquet(f"{DATA}/{split}_transactions.parquet")
    t = t.merge(s[["signal_id", "signal_sanasi"]], on="signal_id")
    v = t.tranzaksiya_vaqti
    burst = ((v.dt.hour == 23) & (v.dt.minute >= 57)) | ((v.dt.hour == 0) & (v.dt.minute == 0) & (v.dt.second == 0))
    t["b"] = np.where(burst, "B", "N")
    t["bin"] = np.digitize(t.miqdor_indeksi, EDGES[1:-1])
    lag = (t.signal_sanasi - v).dt.total_seconds() / 86400
    t["key"] = t.kirim_chiqim.str[:3] + "_" + t.tranzaksiya_turi.str[:4] + "_" + t.b + "_h" + t.bin.astype(str)
    X = pd.crosstab(t.signal_id, t.key)
    tot = t.groupby("signal_id").size()
    # also per type ignoring direction, and shares
    t["key2"] = t.tranzaksiya_turi.str[:4] + "_" + t.b + "_h" + t.bin.astype(str)
    X2 = pd.crosstab(t.signal_id, t.key2)
    S2 = X2.div(tot, axis=0).add_prefix("sh_")
    # recent (last 30 days) type x bin
    r = t[lag <= 30]
    X3 = pd.crosstab(r.signal_id, r.tranzaksiya_turi.str[:4] + "_h" + r.bin.astype(str)).add_prefix("w30_")
    # tiny-card / floor indicators
    tiny = t[(t.tranzaksiya_turi == "karta") & (t.miqdor_indeksi < -2.2)]
    floor = t[(t.tranzaksiya_turi == "bank_otkazmasi") & (t.miqdor_indeksi < -2.8)]
    extra = pd.DataFrame({
        "tiny_card_n": tiny.groupby("signal_id").size(),
        "tiny_card_burst_n": tiny[tiny.b == "B"].groupby("signal_id").size(),
        "tiny_card_min": tiny.groupby("signal_id").miqdor_indeksi.min(),
        "tiny_card_last_lag": (tiny.signal_sanasi - tiny.tranzaksiya_vaqti).dt.total_seconds().groupby(tiny.signal_id).min() / 86400,
        "bank_floor_n": floor.groupby("signal_id").size(),
    })
    X = pd.concat([X.add_prefix("h_"), X2.add_prefix("h2_"), S2, X3, extra], axis=1)
    X = s.set_index("signal_id")[[]].join(X)
    X[[c for c in X.columns if not c.startswith(("sh_", "tiny_card_min", "tiny_card_last"))]] = X[[c for c in X.columns if not c.startswith(("sh_", "tiny_card_min", "tiny_card_last"))]].fillna(0)
    return X.astype("float32")


if __name__ == "__main__":
    for split in ["train", "test"]:
        X = build(split)
        X.to_parquet(f"{DATA}/feat4_{split}.parquet")
        print(split, X.shape)
