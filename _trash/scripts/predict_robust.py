"""Robust ensemble RUL: average the degradation-fraction (lf) across the 4
best-transferring indicators, then map to RUL. Hedges single-indicator risk;
Test1 (all indicators agree clean) is the one high-confidence call.

Indicators (each: lf = (val - train_healthy_median) / (train_EOL_median - healthy), clip[0,1]):
  Spectral_Entropy (est, EOL CV 0.02), Order_BandEnergy (est, 0.02),
  Env_BandEnergy (est, 0.04), OT_RMS (est, 0.41)   -- max over channels, smoothed.
RUL = CAP - mean_lf * (CAP - FLOOR), CAP=53153 (max train RUL@file50), FLOOR=3600 (1h).

Caveat: for Test2/3/5/6 the indicators DISAGREE (Order/Spectral say clean, Env/RMS say
failed); the mean is a central hedge, not a precise estimate. Test1/Test4 are solid.

Run:  python scripts/predict_robust.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
EST = ROOT / "outputs/ot_features/est"; ESTT = ROOT / "outputs/ot_features/test"
OUT = ROOT / "outputs" / "아이사_validation.xlsx"
TESTS = [f"Test{i}" for i in range(1, 7)]
FEATS = ["Spectral_Entropy", "Order_BandEnergy", "Env_BandEnergy", "OT_RMS"]
CAP, FLOOR = 53153.0, 3600.0


def series(which, ident, feat):
    d = EST if which == "train" else ESTT
    fn = f"Train{ident}.csv" if which == "train" else f"{ident}.csv"
    df = pd.read_csv(d / fn).sort_values("File_Index")
    cols = [c for c in df.columns if c.endswith(feat)]
    v = df[cols].to_numpy(float).max(1)
    return pd.Series(v).rolling(5, min_periods=1).median().to_numpy()


def lf(feat, val):
    tr = {t: series("train", t, feat) for t in (1, 2, 3, 4)}
    heal = np.median([np.median(tr[t][:max(3, int(len(tr[t]) * 0.15))]) for t in tr])
    eol = np.median([tr[t][-1] for t in tr])
    return min(max((val - heal) / (eol - heal + 1e-9), 0), 1)


def main():
    rows = []
    for i, n in enumerate(TESTS):
        lfs = [lf(f, series("test", n, f)[-1]) for f in FEATS]
        rul = CAP - np.mean(lfs) * (CAP - FLOOR)
        rows.append({"File": f"Validation{i+1}", "RUL_Score": int(round(max(600, rul))),
                     "_meanlf": round(float(np.mean(lfs)), 2)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df[["File", "RUL_Score"]].to_excel(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
