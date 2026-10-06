"""Causal link-history features for the operation-time rungs (notebooks 12 and 16).

Every feature of a packet uses only earlier packets of the same link; the packet's own RSSI and SNR are never used.
The anchor `a` is the link's mean path loss over the previous hour (fallbacks: the previous 60 packets, then 20), the
target is the deviation y = PL - a, and the history features are expressed relative to the anchor (r_*), so a model
fitted on five links transfers to the sixth.
"""
import numpy as np
import pandas as pd

H1 = ["r_lag1", "r_lag2", "r_lag3", "r_m5", "r_m20", "r_m6h", "r_m24h", "r_m7D", "r_max60", "r_min60", "r_q95", "s20", "s1h", "s24h", "gap", "n1h"]
H2 = H1 + ["r_c1", "r_mc5", "cgap", "frequency"]
H3 = H2 + ["hsin", "hcos", "dow", "work"]
H4 = H3 + ["SF"]
H5 = H4 + ["snr1", "snr20"]
H6 = H5 + ["co2", "humidity", "pm25", "pressure", "temperature", "co2ex"]
H7 = H6 + ["a", "distance", "c_walls", "w_walls"]
RUNGS = {"H1 link history": H1, "H2 + same-channel history": H2, "H3 + clock": H3, "H4 + SF": H4,
         "H5 + SNR of earlier packets": H5, "H6 + environmental sensors": H6, "H7 + link level and geometry": H7}


def build(d, anchor="m1h", tz="Europe/Berlin"):
    """d: one row per packet with device_id, t (UTC datetime), PL, frequency, SF, snr and fold, and co2 when there is a CO2 sensor. Returns a copy
    sorted by link and time with the history features, the anchor `a` (column `anchor`, with fallbacks m60 then m20) and the target y. The clock
    features use the local time zone tz."""
    d = d.sort_values(["device_id", "t"]).reset_index(drop=True)
    g = d.groupby("device_id", sort=False)

    def over_time(col, window, fn):     # rolling window in time, right edge open, per link
        parts = [pd.Series(getattr(x.set_index("t")[col].rolling(window, closed="left"), fn)().values, index=x.index) for _, x in g]
        return pd.concat(parts).sort_index()

    for k in (1, 2, 3):
        d[f"lag{k}"] = g["PL"].shift(k)
    for n in (5, 20, 60):
        d[f"m{n}"] = g["PL"].transform(lambda s: s.shift(1).rolling(n, min_periods=max(2, n // 2)).mean())
    d["s20"] = g["PL"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
    d["max60"] = g["PL"].transform(lambda s: s.shift(1).rolling(60, min_periods=20).max())
    d["min60"] = g["PL"].transform(lambda s: s.shift(1).rolling(60, min_periods=20).min())
    d["q95"] = g["PL"].transform(lambda s: s.shift(1).rolling(1440, min_periods=200).quantile(0.95))
    for w in ("1h", "6h", "24h", "7D"):
        d[f"m{w}"] = over_time("PL", w, "mean")
    d["s1h"] = over_time("PL", "1h", "std")
    d["s24h"] = over_time("PL", "24h", "std")
    d["n1h"] = over_time("PL", "1h", "count")
    d["gap"] = g["t"].diff().dt.total_seconds()

    gc = d.groupby(["device_id", "frequency"], sort=False)      # the same link on the same channel
    d["c1"] = gc["PL"].shift(1)
    d["mc5"] = gc["PL"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    d["cgap"] = gc["t"].diff().dt.total_seconds()

    d["snr1"] = g["snr"].shift(1)                                 # SNR of earlier packets only
    d["snr20"] = g["snr"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    if "co2" in d:
        d["co2ex"] = d["co2"] - g["co2"].transform(lambda s: s.shift(1).rolling(1440, min_periods=200).min())

    local = d["t"].dt.tz_convert(tz)
    hour = local.dt.hour + local.dt.minute / 60
    d["hsin"], d["hcos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    d["dow"] = local.dt.weekday
    d["work"] = ((d["dow"] < 5) & (hour >= 8) & (hour < 18)).astype(int)

    d["a"] = d[anchor].fillna(d["m60"]).fillna(d["m20"])
    for c in ("lag1", "lag2", "lag3", "m5", "m20", "m60", "m1h", "m6h", "m24h", "m7D", "max60", "min60", "q95", "c1", "mc5"):
        d[f"r_{c}"] = d[c] - d["a"]
    d["y"] = d["PL"] - d["a"]
    d["devmean"] = d["device_id"].map(d[d["fold"] != 9].groupby("device_id")["PL"].mean())   # the static per-link reference
    return d
