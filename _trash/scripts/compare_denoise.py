"""Compare HI quality (monotonicity / trendability) ORIGINAL vs DENOISED on the
4 training bearings. The objective guardrail: keep denoising ONLY if HI quality
improves. Uses the same M/T/P-weighted HI as our model.

Run after extract_denoised.py:  python scripts/compare_denoise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ORIG = ROOT / "outputs" / "ot_features" / "est"
DEN = ROOT / "outputs" / "ot_features" / "est_denoised"


def load(folder):
    out = {}
    for tr in (1, 2, 3, 4):
        p = folder / f"Train{tr}.csv"
        if p.exists():
            out[tr] = pd.read_csv(p).sort_values("File_Index").reset_index(drop=True)
    return out


def mono(x):
    d = np.diff(x)
    return abs((np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1))


def trend(x):
    t = np.arange(len(x))
    return abs(np.corrcoef(x, t)[0, 1]) if np.std(x) > 0 else 0.0


def mtp_weights(data):
    cols = [c for c in data[1].columns if c.startswith("Ch")]
    w = {}
    for c in cols:
        mons, trs, eol, rng = [], [], [], []
        for df in data.values():
            x = pd.Series(df[c].to_numpy(float)).rolling(5, min_periods=1).median().to_numpy()
            mons.append(mono(x)); trs.append(trend(x)); eol.append(x[-1]); rng.append(np.ptp(x) + 1e-9)
        prog = float(np.exp(-np.std(eol) / (np.mean(rng) + 1e-9)))
        w[c] = np.mean(mons) + np.mean(trs) + prog
    return w


def build_hi(data, sel, wts):
    # global healthy baseline over selected features
    P = np.vstack([data[t][sel].to_numpy(float)[:max(3, int(len(data[t]) * 0.15))]
                   for t in data])
    mu, sd = P.mean(0), P.std(0) + 1e-9
    his = {}
    for tr, df in data.items():
        z = (df[sel].to_numpy(float) - mu) / sd
        h = (z * wts).sum(1)
        his[tr] = pd.Series(h).rolling(5, min_periods=1).median().cummax().to_numpy()
    return his


def evaluate(data, label):
    w = mtp_weights(data)
    sel = sorted(w, key=lambda c: -w[c])[:10]
    wts = np.array([w[c] for c in sel]); wts = wts / wts.sum()
    his = build_hi(data, sel, wts)
    monos = {tr: mono(his[tr]) for tr in data}
    trends = {tr: trend(his[tr]) for tr in data}
    print(f"[{label}] HI monotonicity: " +
          "  ".join(f"T{tr}={monos[tr]:.3f}" for tr in data) +
          f"  | mean={np.mean(list(monos.values())):.3f}")
    print(f"[{label}] HI trendability: " +
          "  ".join(f"T{tr}={trends[tr]:.3f}" for tr in data) +
          f"  | mean={np.mean(list(trends.values())):.3f}")
    return np.mean(list(monos.values())), np.mean(list(trends.values()))


def main():
    orig, den = load(ORIG), load(DEN)
    if len(den) < 4:
        print(f"denoised features incomplete in {DEN} (have {sorted(den)})")
        return
    print("=== HI quality: ORIGINAL vs DENOISED (training, higher = better) ===\n")
    mo, to = evaluate(orig, "ORIGINAL ")
    print()
    md, td = evaluate(den, "DENOISED ")
    print(f"\n>>> mean monotonicity  {mo:.3f} -> {md:.3f}  ({md-mo:+.3f})")
    print(f">>> mean trendability  {to:.3f} -> {td:.3f}  ({td-to:+.3f})")
    verdict = "KEEP denoising" if (md - mo) > 0.005 or (td - to) > 0.01 else "REJECT (no gain)"
    print(f">>> verdict: {verdict}")


if __name__ == "__main__":
    main()
