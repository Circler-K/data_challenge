"""Is each HI feature's rise REAL degradation, or just the 700-950 rpm cycling?
Regress every feature on RPM (training operation.csv) and compare the feature's
time-trend BEFORE vs AFTER removing the RPM-explained part. A feature whose trend
SURVIVES residualization is trustworthy degradation; one whose trend collapses was
a speed-cycling artifact. (Order tracking should already make OT features robust;
this verifies it.)

Run:  python scripts/rpm_confound.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.operation import load_operation, align_to_vibration, list_vibration_files  # noqa: E402

FO = ROOT / "outputs" / "ot_features" / "faultorder"
FEATS = ["OT_RMS", "Order_Band", "Env_Band", "Fault_BPFO", "Fault_BPFI", "Fault_BSF"]


def trend(x):
    t = np.arange(len(x))
    return abs(np.corrcoef(x, t)[0, 1]) if np.std(x) > 0 else 0.0


def main():
    rows = []
    for tr in (1, 2, 3, 4):
        df = pd.read_csv(FO / f"Train{tr}.csv").sort_values("File_Index").reset_index(drop=True)
        op = load_operation(tr)
        agg = align_to_vibration(op, len(list_vibration_files(tr)))
        rpm = np.array([float(agg.loc[agg.file_idx == k, "rpm_mean"].iloc[0])
                        if (agg.file_idx == k).any() else np.nan
                        for k in range(1, len(df) + 1)])
        # design matrix: rpm + rpm^2 (load/temp omitted; rpm is the cycling driver)
        ok = ~np.isnan(rpm)
        R = np.vstack([np.ones(ok.sum()), rpm[ok], rpm[ok] ** 2]).T
        for c in range(4):
            for f in FEATS:
                col = f"Ch{c}_{f}"
                if col not in df.columns:
                    continue
                y = df[col].to_numpy(float)[ok]
                raw = trend(y)
                beta, *_ = np.linalg.lstsq(R, y, rcond=None)
                resid = y - R @ beta
                res = trend(resid)
                rows.append(dict(feat=f, bearing=tr, raw=raw, resid=res, keep=res / (raw + 1e-9)))
    d = pd.DataFrame(rows)
    print("=== RPM-confound check: trend BEFORE vs AFTER removing rpm(+rpm^2) ===")
    print("(keep ~1.0 => trend is real degradation, NOT speed-cycling)\n")
    g = d.groupby("feat").agg(raw_trend=("raw", "mean"), resid_trend=("resid", "mean"),
                              keep_ratio=("keep", "mean")).sort_values("resid_trend", ascending=False)
    print(g.round(3).to_string())
    print(f"\nmean keep-ratio over all features = {d['keep'].mean():.3f} "
          f"(close to 1.0 => order tracking already removed RPM dependence)")


if __name__ == "__main__":
    main()
