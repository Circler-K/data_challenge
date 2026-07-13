"""Where in life is each of the 6 validation bearings? The whole RUL strategy
hinges on this. For each validation segment (and each training bearing for
reference) compute, against the FIXED training-healthy baseline:

  HI_end       - absolute degradation level at the last timestamp (mean z-score)
  pctile       - where HI_end sits within the training EOL-HI distribution
  slope        - HI trend within the segment (z per hour): RISING => active
                 degradation (knee, predictable RUL); FLAT => pre-knee or
                 stabilised (RUL irreducibly uncertain -> predict conservatively)
  slope_last   - slope over the last half of the segment (recent dynamics)
  mono         - monotonicity of the segment HI

Uses raw mean-z HI (smoothed, NO cummax) so the true rising/flat shape shows.

Run:  python scripts/diagnose_validation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import wiener_rul as W  # noqa: E402

EST = ROOT / "outputs" / "ot_features" / "est"
TEST = ROOT / "outputs" / "ot_features" / "test"
HEALTHY_FRAC = 0.15


FILE_PERIOD = 600


def load(folder, names):
    out = {}
    for n in names:
        p = folder / f"{n}.csv"
        if p.exists():
            df = pd.read_csv(p).sort_values("File_Index").reset_index(drop=True)
            if "t_sec" not in df.columns:
                df["t_sec"] = (df["File_Index"] - 1) * FILE_PERIOD
            out[n] = df
    return out


def baseline(train):
    pools = []
    for df in train.values():
        X = df[W.feat_cols(df)].to_numpy(float)
        pools.append(X[: max(3, int(len(X) * HEALTHY_FRAC))])
    P = np.vstack(pools)
    return P.mean(0), P.std(0) + 1e-9


def hi_raw(df, mu, sd, smooth=5):
    """Smoothed mean-z HI, NO cummax -> shows real rising/flat shape."""
    X = df[W.feat_cols(df)].to_numpy(float)
    z = ((X - mu) / sd).mean(1)
    return pd.Series(z).rolling(smooth, min_periods=1).median().to_numpy()


def slope_per_h(hi, t_sec):
    h = t_sec / 3600.0
    if np.std(h) == 0:
        return 0.0
    return float(np.polyfit(h, hi, 1)[0])


def mono(x):
    d = np.diff(x)
    return abs((np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1))


def describe(name, df, mu, sd, eol_hi):
    hi = hi_raw(df, mu, sd)
    t = df["t_sec"].to_numpy(float)
    dur_h = (t[-1] - t[0]) / 3600.0
    half = len(hi) // 2
    s_all = slope_per_h(hi, t)
    s_last = slope_per_h(hi[half:], t[half:])
    pct = float((np.sum(np.array(eol_hi) <= hi[-1]) / len(eol_hi)) * 100) if eol_hi else np.nan
    return dict(bearing=name, n=len(df), dur_h=round(dur_h, 1),
                hi_start=round(float(hi[0]), 2), hi_end=round(float(hi[-1]), 2),
                slope=round(s_all, 3), slope_last=round(s_last, 3),
                mono=round(mono(hi), 2), pct_of_EOL=round(pct, 0))


def verdict(r, eol_lo):
    """Heuristic classification for the adaptive read-out."""
    rising = r["slope_last"] > 0.15 or r["slope"] > 0.10
    high = r["hi_end"] >= eol_lo * 0.6
    if r["hi_end"] >= eol_lo:
        return "LATE (HI>=train EOL) -> short RUL, predict near-accurate"
    if rising and high:
        return "KNEE (rising + elevated) -> degradation active, RUL predictable"
    if rising:
        return "EARLY-KNEE (rising, still low) -> some signal, moderate RUL"
    return "PRE-KNEE/FLAT (no rise) -> RUL uncertain, predict CONSERVATIVE-short"


def main():
    train = load(EST, [f"Train{i}" for i in (1, 2, 3, 4)])
    test = load(TEST, [f"Test{i}" for i in range(1, 7)])
    mu, sd = baseline(train)

    # training reference: EOL HI (cummax-free end level) + life
    eol_hi, lives = [], []
    print("=== TRAINING reference (full run to failure) ===")
    rows = []
    for n, df in train.items():
        r = describe(n, df, mu, sd, [])
        eol_hi.append(hi_raw(df, mu, sd)[-1]); lives.append(df["t_sec"].iloc[-1] / 3600.0)
        rows.append(r)
    eol_hi_sorted = sorted(eol_hi)
    for r in rows:
        r["pct_of_EOL"] = round(float(np.sum(np.array(eol_hi) <= r["hi_end"]) / len(eol_hi) * 100), 0)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n training EOL HI = {[round(v,1) for v in eol_hi_sorted]}  (min={min(eol_hi):.1f})")
    print(f" training life(h) = {sorted(round(v,1) for v in lives)}\n")

    eol_lo = min(eol_hi)
    print("=== VALIDATION bearings (the 6 segments we must score) ===")
    vrows = [describe(n, df, mu, sd, eol_hi) for n, df in test.items()]
    print(pd.DataFrame(vrows).to_string(index=False))
    print("\n=== per-bearing verdict (drives adaptive conservatism) ===")
    for r in vrows:
        print(f"  {r['bearing']}: HI_end={r['hi_end']:>5}  slope={r['slope']:>6}  "
              f"slope_last={r['slope_last']:>6}  -> {verdict(r, eol_lo)}")

    # ---- degradation-RATE RUL vs the submitted Option-B values ----
    D = float(np.median(eol_hi))                  # threshold = median training EOL HI
    cap_h = float(np.percentile(lives, 25))       # conservative cap = p25 training life
    B = {"Test1": 59351, "Test2": 51138, "Test3": 19936,
         "Test4": 12168, "Test5": 1800, "Test6": 1800}  # submitted Option B (sec)
    print(f"\n=== degradation-RATE RUL  (D=median EOL HI={D:.2f}, cap={cap_h:.1f}h) vs submitted B ===")
    print(f"{'bearing':>8} {'HI_end':>7} {'rate/h':>7} {'rate_RUL_h':>11} {'B_RUL_h':>9}  note")
    for r in vrows:
        rate = r["slope_last"]
        if rate <= 0.02:                          # flat -> degradation not progressing
            rul_h, note = cap_h, "FLAT -> cap (uncertain)"
        else:
            rul_h = float(np.clip((D - r["hi_end"]) / rate, 0.5, cap_h))
            note = "rising -> finite estimate"
        b_h = B[r["bearing"]] / 3600.0
        print(f"{r['bearing']:>8} {r['hi_end']:>7.2f} {rate:>7.3f} {rul_h:>11.1f} {b_h:>9.1f}  {note}")


if __name__ == "__main__":
    main()
