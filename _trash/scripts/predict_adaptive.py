"""TODAY's submission: adaptive-conservative ensemble.
- Robust base = mean lf over 4 best-transferring indicators (Spectral/Order/Env/RMS) -> RUL.
- Per-bearing CONFIDENCE = std of the 4 indicator lf values (low std = indicators agree = confident).
- Conservative factor = 1 - 0.35*min(std/0.5, 1): confident bearings (std~0) keep ~full RUL;
  uncertain bearings (indicators disagree) are trimmed shorter (asymmetric metric: over ÷30 > under ÷50,
  and recent challenges show over-prediction is the dominant error for defect-progressing/censored units).
  0.35 coefficient = MILD trim (not halving) since censoring implies true RUL is "longer than observed".

Run:  python scripts/predict_adaptive.py
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
CAP, FLOOR, COEF = 53153.0, 3600.0, 0.35


def series(which, ident, feat):
    d = EST if which == "train" else ESTT
    fn = f"Train{ident}.csv" if which == "train" else f"{ident}.csv"
    df = pd.read_csv(d / fn).sort_values("File_Index")
    cols = [c for c in df.columns if c.endswith(feat)]
    return pd.Series(df[cols].to_numpy(float).max(1)).rolling(5, min_periods=1).median().to_numpy()


def lf(feat, val):
    tr = {t: series("train", t, feat) for t in (1, 2, 3, 4)}
    heal = np.median([np.median(tr[t][:max(3, int(len(tr[t]) * 0.15))]) for t in tr])
    eol = np.median([tr[t][-1] for t in tr])
    return min(max((val - heal) / (eol - heal + 1e-9), 0), 1)


def main():
    rows = []
    for i, n in enumerate(TESTS):
        lfs = [lf(f, series("test", n, f)[-1]) for f in FEATS]
        base = CAP - np.mean(lfs) * (CAP - FLOOR)
        std = np.std(lfs)
        factor = 1 - COEF * min(std / 0.5, 1)
        rul = max(600, base * factor)
        rows.append({"File": f"Validation{i+1}", "RUL_Score": int(round(rul)),
                     "_std": round(float(std), 2), "_factor": round(float(factor), 2)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df[["File", "RUL_Score"]].to_excel(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
