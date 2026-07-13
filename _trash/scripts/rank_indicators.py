"""Rigorous, HONEST ranking of every candidate degradation indicator for the
degradation-fraction RUL model. Avoids the crest quick-check artifact by using
FULL-data anchors and proper cut-40/50/60 LOO. For each indicator reports:
  EOL_CV  : std/mean of the last-file value across the 4 training bearings (transfer; lower=better)
  LOO     : cut-40/50/60 leave-one-bearing-out /30 score (test-faithful)
  implB   : implied-B of its 6 test predictions vs B=0.49 (consistency w/ real anchor)
Ranks by LOO. Then builds the best indicator's test vector + objective verdict.

Run:  python scripts/rank_indicators.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import sys
ROOT = Path("c:/Users/User/WorkSpace/data_challenge"); sys.path.insert(0, str(ROOT))
from src.scoring import a_rul_score  # noqa: E402

STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}; FP = 600; CUTS = (40, 50, 60)
TESTS = [f"Test{i}" for i in range(1, 7)]; CAP, FLOOR = 53153.0, 3600.0
B = [59351, 51138, 19936, 12168, 1800, 1800]
EST = ROOT / "outputs/ot_features/est"; ESTT = ROOT / "outputs/ot_features/test"
SF = ROOT / "outputs/ot_features/subfile"; SFT = SF / "test"

# indicator: (source, column-suffix or exact). est cols are per-channel (max); subfile are single.
EST_FEATS = ["OT_RMS", "OT_Kurtosis", "OT_CrestFactor", "Order_BandEnergy", "Spectral_Entropy", "Env_BandEnergy"]
SF_FEATS = ["crest_max", "nonstat", "kurt_mean"]


def series(which, ident, feat, src):
    if src == "est":
        d = EST if which == "train" else ESTT
        fn = f"Train{ident}.csv" if which == "train" else f"{ident}.csv"
        df = pd.read_csv(d / fn).sort_values("File_Index")
        cols = [c for c in df.columns if c.endswith(feat)]
        v = df[cols].to_numpy(float).max(1)
    else:  # subfile
        d = SF if which == "train" else SFT
        fn = f"Train{ident}.csv" if which == "train" else f"{ident}.csv"
        df = pd.read_csv(d / fn).sort_values("File_Index")
        v = df[feat].to_numpy(float)
    return pd.Series(v).rolling(5, min_periods=1).median().to_numpy()


def lf_pred(feat, src, val, anchor_bearings):
    tr = {t: series("train", t, feat, src) for t in anchor_bearings}
    heal = np.median([np.median(tr[t][:max(3, int(len(tr[t]) * 0.15))]) for t in anchor_bearings])
    eol = np.median([tr[t][-1] for t in anchor_bearings])
    lf = min(max((val - heal) / (eol - heal + 1e-9), 0), 1)
    return CAP - lf * (CAP - FLOOR)


def evaluate(feat, src):
    tr = {t: series("train", t, feat, src) for t in (1, 2, 3, 4)}
    eolv = np.array([tr[t][-1] for t in (1, 2, 3, 4)])
    cv = abs(eolv.std() / (eolv.mean() + 1e-9))
    # LOO
    scs = []
    for cut in CUTS:
        for held in (1, 2, 3, 4):
            others = [t for t in (1, 2, 3, 4) if t != held]
            pred = lf_pred(feat, src, tr[held][cut - 1], others)
            act = STOP[held] - ((cut - 1) * FP + 60)
            scs.append(float(a_rul_score(act, max(600, pred))))
    loo = np.mean(scs)
    # test preds + implied-B
    preds = [lf_pred(feat, src, series("test", n, feat, src)[-1], [1, 2, 3, 4]) for n in TESTS]
    imp = float(np.mean([float(a_rul_score(preds[i], B[i])) for i in range(6)]))
    return cv, loo, imp, [int(round(p)) for p in preds]


def main():
    rows = []
    for f in EST_FEATS:
        cv, loo, imp, pr = evaluate(f, "est"); rows.append((f, cv, loo, imp, pr))
    for f in SF_FEATS:
        cv, loo, imp, pr = evaluate(f, "sf"); rows.append((f, cv, loo, imp, pr))
    rows.sort(key=lambda r: -r[2])  # by LOO
    print(f"{'indicator':>16}{'EOL_CV':>8}{'LOO/30':>8}{'implB':>7}   test preds")
    for f, cv, loo, imp, pr in rows:
        print(f"{f:>16}{cv:>8.2f}{loo:>8.3f}{imp:>7.3f}   {pr}")
    best = rows[0]
    print(f"\n>>> best by LOO: {best[0]}  (LOO {best[2]:.3f}, implB {best[3]:.3f}, EOL_CV {best[1]:.2f})")
    print(f"    vector: {best[4]}")


if __name__ == "__main__":
    main()
