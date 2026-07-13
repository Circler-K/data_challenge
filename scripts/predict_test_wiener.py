"""Generate preliminary-submission RUL predictions for the 6 validation bearings
(Test1..Test6) using the selected OT-HI -> Wiener first-passage-time model.

For each validation bearing (a continuous 50-file segment cut BEFORE failure):
  1. estimate RPM per file from vibration (rpm_estimator), order-track, extract
     the same OT features used in training,
  2. build the monotone HI (own first-15% window as healthy reference — NOTE the
     segment starts mid-life, so this is a relative, not absolute, baseline),
  3. Wiener FHT from the HI at the LAST file to a cross-bearing threshold D
     (calibrated from the 4 training bearings' end-of-life HI), conservative
     read-out, clamped to [floor, absolute cap].

Outputs: outputs/test_rul_wiener.csv and a draft outputs/submission_validation.xlsx
(REVIEW before uploading — confirm the official template columns + team name).

Run:  python scripts/predict_test_wiener.py
"""
from __future__ import annotations

import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
INHWAN = Path("c:/Users/User/WorkSpace/INHWAN")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(INHWAN))

import rpm_estimator as R                       # noqa: E402
from ot_rpm_impact import order_track, ot_features, CHS  # noqa: E402
import wiener_rul as W  # noqa: E402
from wiener_rul import wiener_rul, load as load_train_feats  # noqa: E402

TEST_ROOT = ROOT / "Test"
BEARINGS = ("Test1", "Test2", "Test3", "Test4", "Test5", "Test6")
FILE_PERIOD = 600
WINDOW, PCT, FLOOR_H = 10, 0.5, 0.5
TEAM = "아이사"   # submission file -> {TEAM}_validation.xlsx


def bearing_files(name: str) -> list[Path]:
    """Resolve a validation bearing's 50 TDMS files despite messy nesting:
    pick, among all dirs ending in `name`, the one holding the most .tdms."""
    paths = glob.glob(str(TEST_ROOT / "**" / name / "*.tdms"), recursive=True)
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_dir[os.path.dirname(p)].append(p)
    if not by_dir:
        return []
    best = max(by_dir.values(), key=len)
    return [Path(p) for p in sorted(best)]


TEST_FEAT = ROOT / "outputs" / "ot_features" / "test"
HEALTHY_FRAC = 0.15


def extract_bearing(name: str) -> pd.DataFrame:
    """Extract OT features for one validation bearing; cache to CSV so HI/model
    iteration needs no re-extraction."""
    TEST_FEAT.mkdir(parents=True, exist_ok=True)
    cache = TEST_FEAT / f"{name}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    files = bearing_files(name)
    if not files:
        raise FileNotFoundError(f"no TDMS found for {name} under {TEST_ROOT}")
    # RPM per file: harmonic-sum then stepwise refine (vibration-only)
    ch0, sig_cache = [], []
    for f in files:
        chs = [c[:] for c in TdmsFile.read(str(f)).groups()[0].channels()][:4]
        sig_cache.append(chs)
        ch0.append(chs[0])
    est_rpm = R.refine_stepwise(R.estimate_rpm_series(ch0))
    rows = []
    for k, chs in enumerate(sig_cache):
        row = {"File_Index": k + 1}
        for ch in CHS:
            if ch < len(chs):
                row.update(ot_features(order_track(chs[ch], float(est_rpm[k])), ch))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["t_sec"] = (df["File_Index"] - 1) * FILE_PERIOD
    df.to_csv(cache, index=False)
    return df


def global_baseline(train: dict) -> tuple[np.ndarray, np.ndarray]:
    """Healthy reference (mu, sigma) pooled from the first `HEALTHY_FRAC` of
    every TRAINING bearing — an ABSOLUTE healthy level. Validation segments
    start mid-life so they have no healthy window of their own; this lets their
    HI reflect true degradation against a fixed reference (same test rig)."""
    pools = []
    for df in train.values():
        cols = W.feat_cols(df)
        X = df[cols].to_numpy(float)
        n_h = max(3, int(len(X) * HEALTHY_FRAC))
        pools.append(X[:n_h])
    P = np.vstack(pools)
    return P.mean(axis=0), P.std(axis=0) + 1e-9


def hi_global(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray,
              smooth: int = 5) -> np.ndarray:
    """Monotone HI vs a FIXED (training-healthy) baseline."""
    X = df[W.feat_cols(df)].to_numpy(float)
    z = ((X - mu) / sigma).mean(axis=1)
    return pd.Series(z).rolling(smooth, min_periods=1).median().cummax().to_numpy()


def main():
    train = load_train_feats("est")
    mu, sigma = global_baseline(train)
    eol_hi = sorted(float(hi_global(df, mu, sigma)[-1]) for df in train.values())
    eol_sec = sorted(float(df["t_sec"].iloc[-1] + 60) for df in train.values())
    D = float(np.percentile(eol_hi, 25))
    abs_cap = float(np.percentile(eol_sec, 25))   # no elapsed term: validation age unknown
    print(f"GLOBAL-baseline calibration: training EOL HI={[round(v,1) for v in eol_hi]} "
          f"-> D(p25)={D:.2f}")
    print(f"  training life(h)={[round(v/3600,1) for v in eol_sec]} -> "
          f"abs_cap(p25)={abs_cap/3600:.1f}h\n")

    out = []
    for name in BEARINGS:
        t0 = time.time()
        df = extract_bearing(name)
        hi = hi_global(df, mu, sigma)
        t = df["t_sec"].to_numpy()
        rul = wiener_rul(hi, t, D, WINDOW, PCT, cap=abs_cap, floor_sec=FLOOR_H * 3600)
        out.append(dict(bearing=name, n_files=len(df),
                        hi_last=float(hi[-1]),
                        rul_sec=round(rul, 1), rul_h=round(rul / 3600, 3)))
        print(f"  {name}: n={len(df)} HI_last={hi[-1]:6.2f}  "
              f"RUL={rul/3600:6.2f}h ({rul:,.0f}s)  [{time.time()-t0:.0f}s]")

    res = pd.DataFrame(out)
    res.to_csv(ROOT / "outputs" / "test_rul_wiener.csv", index=False)
    # Official template (per organizer example): columns File | RUL_Score,
    # rows ValidationN, RUL in seconds. We map Test{i} -> Validation{i}.
    # NOTE: organizer example shows Validation1..11; we only have 6 Test folders.
    # Confirm the true validation-bearing count before uploading.
    # Row labels MUST match the organizer's actual template (Test1..Test6).
    sub = pd.DataFrame({
        "File": list(res["bearing"]),
        "RUL_Score": res["rul_sec"].round().astype(int),
    })
    try:
        sub.to_excel(ROOT / "outputs" / f"{TEAM}_validation.xlsx", index=False)
        print(f"\nsaved outputs/test_rul_wiener.csv + {TEAM}_validation.xlsx")
    except Exception as e:
        print(f"\nsaved test_rul_wiener.csv (xlsx skipped: {e})")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
