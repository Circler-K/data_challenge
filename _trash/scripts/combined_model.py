"""Combined model: ensemble the best-transferring degradation indicators, treat the
spread as uncertainty, and make an asymmetric-metric-conservative (RULSurv-style
probabilistic) decision.

Indicators (each -> lf degradation-fraction RUL, anchored to training healthy/EOL):
  crest_max        (sub-file peak crest factor) — EOL transfer CV 0.20 (best impulsiveness)
  Spectral_Entropy (est)                         — EOL transfer CV 0.02 (best overall)
  OT_RMS           (faultorder)                  — CV 0.41 (energy, weaker transfer)

Per bearing: the 3 RUL estimates = our belief spread. Decision rules evaluated by
cut-50 LOO (/30): median / mean / conservative-min-lean / transfer-weighted. The
asymmetric metric (over ÷30 > under ÷50) favors leaning to the lower (conservative) end.

Run:  python scripts/combined_model.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import sys
ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
from src.scoring import a_rul_score  # noqa: E402

STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}; FP = 600; CUTS = (40, 50, 60)
TESTS = [f"Test{i}" for i in range(1, 7)]
CAP, FLOOR = 53153.0, 3600.0
F = ROOT / "outputs" / "ot_features"
# (folder, filename-pattern for train, test subdir, column)
SRC = {
    "crest_max": (F / "subfile", F / "subfile" / "test", "crest_max"),
    "Spectral":  (F / "est",     F / "test",             "Spectral_Entropy_max"),  # placeholder, handled below
    "RMS":       (F / "faultorder", F / "faultorder" / "test", "OT_RMS_max"),
}


def load_ind(name, which, key):
    """Return {id: smoothed indicator series}. which='train'|'test'."""
    out = {}
    ids = [1, 2, 3, 4] if which == "train" else TESTS
    for i in ids:
        if name == "crest_max":
            d = F / "subfile" if which == "train" else F / "subfile" / "test"
            fn = f"Train{i}.csv" if which == "train" else f"{i}.csv"
            df = pd.read_csv(d / fn); v = df["crest_max"].to_numpy(float)
        elif name == "Spectral":
            d = F / "est" if which == "train" else F / "test"
            fn = f"Train{i}.csv" if which == "train" else f"{i}.csv"
            df = pd.read_csv(d / fn).sort_values("File_Index")
            cols = [c for c in df.columns if c.endswith("Spectral_Entropy")]
            v = df[cols].to_numpy(float).max(1)
        else:  # RMS
            d = F / "faultorder" if which == "train" else F / "faultorder" / "test"
            fn = f"Train{i}.csv" if which == "train" else f"{i}.csv"
            df = pd.read_csv(d / fn).sort_values("File_Index")
            cols = [c for c in df.columns if c.endswith("OT_RMS")]
            v = df[cols].to_numpy(float).max(1)
        out[i] = pd.Series(v).rolling(5, min_periods=1).median().to_numpy()
    return out


INDS = ["crest_max", "Spectral", "RMS"]
TR = {n: load_ind(n, "train", None) for n in INDS}
TE = {n: load_ind(n, "test", None) for n in INDS}


def lf_rul(name, series_val, bearings_for_anchor):
    heal = np.median([np.median(TR[name][t][:max(3, int(len(TR[name][t]) * 0.15))]) for t in bearings_for_anchor])
    eol = np.median([TR[name][t][-1] for t in bearings_for_anchor])
    lf = min(max((series_val - heal) / (eol - heal + 1e-9), 0), 1)
    return CAP - lf * (CAP - FLOOR)


def ensemble_estimates(idx, which, cut=None, anchor_bearings=None):
    """3 indicator RUL estimates for a bearing."""
    ests = []
    for name in INDS:
        s = (TE[name][idx] if which == "test" else TR[name][idx])
        val = s[cut - 1] if cut else s[-1]
        ests.append(lf_rul(name, val, anchor_bearings))
    return np.array(ests)


def decide(ests, rule):
    if rule == "median": return np.median(ests)
    if rule == "mean": return np.mean(ests)
    if rule == "min": return np.min(ests)
    if rule == "q35": return np.percentile(ests, 35)
    if rule == "consv": return np.median(ests) * 0.85  # median, mild conservative shrink


def main():
    rules = ["median", "mean", "q35", "consv", "min"]
    print("=== cut-50 LOO (/30) per decision rule ===")
    best_rule, best = None, -1
    for rule in rules:
        scs = []
        for cut in CUTS:
            for held in [1, 2, 3, 4]:
                others = [t for t in [1, 2, 3, 4] if t != held]
                ests = ensemble_estimates(held, "train", cut=cut, anchor_bearings=others)
                pred = decide(ests, rule)
                act = STOP[held] - ((cut - 1) * FP + 60)
                scs.append(float(a_rul_score(act, max(600, pred))))
        m = np.mean(scs)
        if m > best: best, best_rule = m, rule
        print(f"  {rule:>8}: {m:.3f}")
    print(f"  >> best rule = {best_rule} ({best:.3f})")

    # test predictions with best rule
    B = [59351, 51138, 19936, 12168, 1800, 1800]
    print(f"\n=== test predictions (rule={best_rule}) ===")
    print(f"{'bear':>6}{'crest':>8}{'Spectral':>9}{'RMS':>8}{'->RUL':>8}")
    preds = []
    for i, n in enumerate(TESTS):
        ests = ensemble_estimates(n, "test", anchor_bearings=[1, 2, 3, 4])
        p = max(600, decide(ests, best_rule)); preds.append(p)
        print(f"{n:>6}{ests[0]:>8.0f}{ests[1]:>9.0f}{ests[2]:>8.0f}{p:>8.0f}")
    imp = float(np.mean([float(a_rul_score(preds[i], B[i])) for i in range(6)]))
    print(f"\nvector: {[int(round(p)) for p in preds]}")
    print(f"implied-B = {imp:.3f} (real 0.49)")


if __name__ == "__main__":
    main()
