"""Test-FAITHFUL validation: the real validation bearings give 50 files each, so
mirror that exactly -> cut each training bearing at file 50, predict RUL from only
those 50 files, score with the OFFICIAL /30 A_RUL against the true RUL
(= operation torque-stop time - 50th-file measurement-end, the gap included).

Compares candidate RUL strategies under leave-one-bearing-out, then sweeps a single
global conservative multiplier to find what the asymmetric metric actually rewards.

Run:  python scripts/cut50_harness.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.scoring import a_rul_score  # noqa: E402

FO = ROOT / "outputs" / "ot_features" / "faultorder"
CUT = 50
FILE_PERIOD = 600
MEAS_END_50 = (CUT - 1) * FILE_PERIOD + 60  # 29460 s
STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}  # operation torque-stop times (s)
HEALTHY = 0.15
# HI features: BOTH general band + fault order (won the HI-quality compare, marginally)
HI_FEATS = ["Order_Band", "Env_Band", "Fault_BPFO", "Fault_BPFI", "Fault_BSF"]


def load():
    d = {}
    for tr in (1, 2, 3, 4):
        df = pd.read_csv(FO / f"Train{tr}.csv").sort_values("File_Index").reset_index(drop=True)
        d[tr] = df
    return d


def cols(df):
    return [f"Ch{c}_{f}" for c in range(4) for f in HI_FEATS if f"Ch{c}_{f}" in df.columns]


def baseline(train):
    pools = []
    for df in train.values():
        X = df[cols(df)].to_numpy(float)
        pools.append(X[: max(3, int(len(X) * HEALTHY))])
    P = np.vstack(pools)
    return P.mean(0), P.std(0) + 1e-9


def hi_of(df, mu, sd):
    X = df[cols(df)].to_numpy(float)
    z = ((X - mu) / sd).mean(1)
    return pd.Series(z).rolling(5, min_periods=1).median().cummax().to_numpy()


def actual_rul(tr):
    return STOP[tr] - MEAS_END_50


def total_life(tr):
    return STOP[tr]


def predict(method, held, train, mu, sd):
    """Predict RUL (sec) for held-out bearing using only its first 50 files."""
    others = [t for t in train if t != held]
    hi_full = {t: hi_of(train[t], mu, sd) for t in train}
    hi_h = hi_full[held][:CUT]
    other_rul = [actual_rul(t) for t in others]
    other_life = [total_life(t) for t in others]

    if method == "const_median":
        return float(np.median(other_rul))
    if method == "meanlife_elapsed":
        return float(np.median(other_life) - MEAS_END_50)
    if method == "rate":
        # threshold D = median EOL HI of others; slope over last 20 of the 50 files
        D = float(np.median([hi_full[t][-1] for t in others]))
        h = hi_h
        w = min(20, len(h))
        x = np.arange(w) * FILE_PERIOD
        slope = np.polyfit(x, h[-w:], 1)[0]  # HI per sec
        if slope <= 1e-9:
            return float(np.median(other_rul))      # flat -> fall back to median
        rul = (D - h[-1]) / slope
        return float(np.clip(rul, 600, 1.2 * max(other_life)))
    if method == "lifefrac":
        # life fraction = where others first reach hi_h[-1], then RUL=(1-lf)*median_life
        cur = hi_h[-1]; fracs = []
        for t in others:
            ht = hi_full[t]; idx = np.argmax(ht >= cur)
            if ht[idx] >= cur:
                fracs.append(idx / len(ht))
            else:
                fracs.append(1.0)
        lf = float(np.median(fracs))
        return float(max(600, (1 - lf) * np.median(other_life)))
    raise ValueError(method)


def main():
    train = load()
    mu, sd = baseline(train)
    methods = ["const_median", "meanlife_elapsed", "lifefrac", "rate"]
    print(f"actual RUL at file 50 (s): " +
          ", ".join(f"T{t}={actual_rul(t)}" for t in train))
    print(f"{'method':>16} | " + " ".join(f"T{t:>1}sc" for t in train) +
          " | mean@f=1.0   best_f  best_mean")
    for m in methods:
        preds = {t: predict(m, t, train, mu, sd) for t in train}
        base_scores = [float(a_rul_score(actual_rul(t), preds[t])) for t in train]
        # sweep one global conservative multiplier
        best_f, best_mean = 1.0, np.mean(base_scores)
        for f in np.round(np.arange(0.3, 1.31, 0.05), 2):
            sc = np.mean([float(a_rul_score(actual_rul(t), preds[t] * f)) for t in train])
            if sc > best_mean:
                best_mean, best_f = sc, f
        sc_str = " ".join(f"{s:>4.2f}" for s in base_scores)
        print(f"{m:>16} | {sc_str} | {np.mean(base_scores):>8.3f}   {best_f:>5.2f}  {best_mean:>8.3f}")

    print("\nper-method predictions (s, f=1.0):")
    for m in methods:
        preds = {t: predict(m, t, train, mu, sd) for t in train}
        print(f"  {m:>16}: " + " ".join(f"T{t}={preds[t]:>7.0f}" for t in train))


if __name__ == "__main__":
    main()
