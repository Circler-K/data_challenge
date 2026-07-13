"""Comprehensive HI-quality grid search (no leaderboard): factorially compare
HI construction choices on the 4 training bearings and report the most promising
by monotonicity / trendability (and validation-bearing discrimination).

Axes:
  - normalization: global-healthy baseline   vs   per-RPM-regime baseline
  - weighting:     equal mean of z-scores     vs   M/T/P-weighted top-10
  - (denoising handled separately by extract_denoised.py; pass --src est_denoised)

RPM regime (700 vs 950) from operation.csv (training truth); removes the
regime-induced feature jumps that are a confound, not degradation.

Run:  python scripts/hi_grid_compare.py [--src est|est_denoised]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.operation import load_operation, align_to_vibration, list_vibration_files  # noqa: E402

HEALTHY_FRAC = 0.15
RPM_THRESH = 850.0


def load(src):
    base = ROOT / "outputs" / "ot_features" / src
    out = {}
    for tr in (1, 2, 3, 4):
        df = pd.read_csv(base / f"Train{tr}.csv").sort_values("File_Index").reset_index(drop=True)
        # attach true RPM per file (regime label)
        op = load_operation(tr)
        agg = align_to_vibration(op, len(list_vibration_files(tr)))
        rpm = {int(r.file_idx): float(r.rpm_mean) for r in agg.itertuples()}
        df["rpm"] = df["File_Index"].map(rpm)
        df["regime"] = np.where(df["rpm"].to_numpy() < RPM_THRESH, "lo", "hi")
        out[tr] = df
    return out


def mono(x):
    d = np.diff(x); return abs((np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1))


def trend(x):
    t = np.arange(len(x)); return abs(np.corrcoef(x, t)[0, 1]) if np.std(x) > 0 else 0.0


def feat_cols(df):
    return [c for c in df.columns if c.startswith("Ch")]


def mtp_weights(data):
    cols = feat_cols(data[1])
    w = {}
    for c in cols:
        mons, trs, eol, rng = [], [], [], []
        for df in data.values():
            x = pd.Series(df[c].to_numpy(float)).rolling(5, min_periods=1).median().to_numpy()
            mons.append(mono(x)); trs.append(trend(x)); eol.append(x[-1]); rng.append(np.ptp(x) + 1e-9)
        prog = float(np.exp(-np.std(eol) / (np.mean(rng) + 1e-9)))
        w[c] = np.mean(mons) + np.mean(trs) + prog
    return w


def baseline_global(data, sel):
    P = np.vstack([data[t][sel].to_numpy(float)[:max(3, int(len(data[t]) * HEALTHY_FRAC))]
                   for t in data])
    return P.mean(0), P.std(0) + 1e-9


def hi_for(data, sel, wts, regime_norm):
    """Return {tr: hi}. regime_norm=True => z-score each file against the healthy
    baseline of ITS OWN rpm regime (removes 700/950 jumps)."""
    his = {}
    if not regime_norm:
        mu, sd = baseline_global(data, sel)
    else:
        # per-regime global healthy baseline pooled across bearings
        reg_stats = {}
        for r in ("lo", "hi"):
            pools = []
            for t in data:
                df = data[t]
                m = (df["regime"] == r).to_numpy()
                nh = max(3, int(m.sum() * HEALTHY_FRAC))
                if m.sum() >= 3:
                    pools.append(df.loc[m, sel].to_numpy(float)[:nh])
            P = np.vstack(pools)
            reg_stats[r] = (P.mean(0), P.std(0) + 1e-9)
    for tr, df in data.items():
        X = df[sel].to_numpy(float)
        if not regime_norm:
            z = (X - mu) / sd
        else:
            z = np.zeros_like(X)
            for r in ("lo", "hi"):
                m = (df["regime"] == r).to_numpy()
                mu_r, sd_r = reg_stats[r]
                z[m] = (X[m] - mu_r) / sd_r
        h = (z * wts).sum(1)
        his[tr] = pd.Series(h).rolling(5, min_periods=1).median().cummax().to_numpy()
    return his


def evaluate(data, weighting, regime_norm):
    w = mtp_weights(data)
    if weighting == "mtp":
        sel = sorted(w, key=lambda c: -w[c])[:10]
        wts = np.array([w[c] for c in sel]); wts = wts / wts.sum()
    else:  # equal mean over the original HI feature set
        sel = [f"Ch{c}_{f}" for c in range(4) for f in ("OT_RMS", "Order_BandEnergy", "Env_BandEnergy")]
        wts = np.ones(len(sel)) / len(sel)
    his = hi_for(data, sel, wts, regime_norm)
    m = np.mean([mono(his[tr]) for tr in data])
    t = np.mean([trend(his[tr]) for tr in data])
    return m, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="est")
    args = ap.parse_args()
    data = load(args.src)
    print(f"=== HI-quality grid (src={args.src}, training; higher=better) ===")
    print(f"{'weighting':>10} {'regime_norm':>12} {'monotonic':>10} {'trend':>8} {'sum':>8}")
    rows = []
    for weighting in ("equal", "mtp"):
        for regime_norm in (False, True):
            m, t = evaluate(data, weighting, regime_norm)
            rows.append((weighting, regime_norm, m, t, m + t))
            print(f"{weighting:>10} {str(regime_norm):>12} {m:>10.3f} {t:>8.3f} {m+t:>8.3f}")
    best = max(rows, key=lambda r: r[4])
    print(f"\n>>> BEST: weighting={best[0]}, regime_norm={best[1]}  "
          f"(monotonic={best[2]:.3f}, trend={best[3]:.3f})")


if __name__ == "__main__":
    main()
