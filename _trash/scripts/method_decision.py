"""DEFINITIVE method comparison for the competition. Test-faithful cut-based LOO
(cuts 40/50/60 -> 12 eval points instead of 4, less noisy), scored with the
OFFICIAL /30 A_RUL. Each method gets its best GLOBAL conservative factor f; the
HI-scaling method also gets a per-bearing strength alpha so we can SEE whether
per-bearing adaptation actually beats a robust near-constant.

  pred_i = base * f * (1 - alpha * z_i)      z_i = standardized current HI of bearing i
  alpha=0 -> pure constant ;  alpha>0 -> shorter RUL for more-degraded bearings

Picks the winner, then applies it to the 6 validation bearings (cut=50) -> final
submission vector.

Run:  python scripts/method_decision.py
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
FILE_PERIOD = 600
STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}
CUTS = (40, 50, 60)
HEALTHY = 0.15
HI_FEATS = ["Order_Band", "Env_Band", "Fault_BPFO", "Fault_BPFI", "Fault_BSF"]
TESTS = [f"Test{i}" for i in range(1, 7)]


def load(folder, names, prefix):
    d = {}
    for n in names:
        p = folder / f"{prefix}{n}.csv" if prefix else folder / f"{n}.csv"
        d[n] = pd.read_csv(p).sort_values("File_Index").reset_index(drop=True)
    return d


def cols(df):
    return [f"Ch{c}_{f}" for c in range(4) for f in HI_FEATS if f"Ch{c}_{f}" in df.columns]


def baseline(train):
    P = np.vstack([train[t][cols(train[t])].to_numpy(float)[:max(3, int(len(train[t]) * HEALTHY))]
                   for t in train])
    return P.mean(0), P.std(0) + 1e-9


def hi(df, mu, sd):
    z = ((df[cols(df)].to_numpy(float) - mu) / sd).mean(1)
    return pd.Series(z).rolling(5, min_periods=1).median().cummax().to_numpy()


def actual_rul(tr, cut):
    return STOP[tr] - ((cut - 1) * FILE_PERIOD + 60)


def eval_method(train, mu, sd, alpha):
    """Return list of (actual, base_pred, z) over all (cut, held) LOO points."""
    pts = []
    hi_full = {t: hi(train[t], mu, sd) for t in train}
    for cut in CUTS:
        for held in train:
            others = [t for t in train if t != held]
            base = float(np.median([actual_rul(t, cut) for t in others]))
            hcut_held = hi_full[held][cut - 1]
            hcut_others = np.array([hi_full[t][cut - 1] for t in others])
            mu_h, sd_h = hcut_others.mean(), hcut_others.std() + 1e-9
            z = (hcut_held - mu_h) / sd_h
            pred = base * (1 - alpha * z)
            pts.append((actual_rul(held, cut), max(600.0, pred), z))
    return pts


def best_f(pts):
    best, bf = -1, 1.0
    for f in np.round(np.arange(0.4, 1.21, 0.05), 2):
        sc = np.mean([float(a_rul_score(a, p * f)) for a, p, _ in pts])
        if sc > best:
            best, bf = sc, f
    return bf, best


def main():
    train = load(FO, [1, 2, 3, 4], "Train")
    mu, sd = baseline(train)

    print("=== DEFINITIVE method comparison (cuts 40/50/60 LOO, official /30) ===\n")
    print(f"{'method':>22} {'best_f':>7} {'mean A_RUL':>11}")
    results = {}
    # constant (alpha=0) and HI-scaled with several alphas
    for alpha in (0.0, 0.15, 0.30, 0.50, 0.80):
        pts = eval_method(train, mu, sd, alpha)
        bf, sc = best_f(pts)
        name = "const (alpha=0)" if alpha == 0 else f"HI-scaled a={alpha}"
        results[alpha] = (bf, sc)
        print(f"{name:>22} {bf:>7.2f} {sc:>11.3f}")

    best_alpha = max(results, key=lambda a: results[a][1])
    bf, sc = results[best_alpha]
    print(f"\n>>> WINNER: alpha={best_alpha} (per-bearing strength), f={bf}, LOO A_RUL={sc:.3f}")
    print(f"    (alpha=0 => constant is best; alpha>0 => per-bearing HI scaling helps)")

    # apply winner to the 6 validation bearings (cut=50)
    test = load(FO / "test", TESTS, "")
    hi_full = {t: hi(train[t], mu, sd) for t in train}
    base = float(np.median([actual_rul(t, 50) for t in train]))
    hcut_tr = np.array([hi_full[t][49] for t in train])
    mu_h, sd_h = hcut_tr.mean(), hcut_tr.std() + 1e-9
    print(f"\n=== final test predictions (winner alpha={best_alpha}, f={bf}, base={base:.0f}s) ===")
    sub = {}
    for n in TESTS:
        h = hi(test[n], mu, sd)[49]
        z = (h - mu_h) / sd_h
        pred = max(600.0, base * (1 - best_alpha * z) * bf)
        sub[n] = int(round(pred))
        print(f"  {n}: HI50={h:>6.2f}  z={z:>6.2f}  RUL={sub[n]:>7d}s ({sub[n]/3600:.1f}h)")
    print("\nsubmission vector (s):", sub)


if __name__ == "__main__":
    main()
