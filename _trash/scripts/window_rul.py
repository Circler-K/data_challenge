"""IMPROVED model: windowed trajectory matching.

Problem found: at a single fixed cut, HI vs RUL is confounded by total-life
differences ACROSS bearings (corr came out +0.87, the wrong sign). But WITHIN
one bearing's trajectory, more degradation => less RUL (the correct, intuitive
relation). Fix: slide a 50-file window along every training trajectory; each
window end e gives one labelled sample (window-shape features -> RUL at e). This
(a) multiplies 4 trajectories into hundreds of samples, (b) learns the within-
trajectory degradation->RUL map, (c) sidesteps the cross-bearing life confound.

Validate: LOO by bearing (train on 3 bearings' windows, predict held bearing's
first-50 window = the exact test scenario), score /30; plus consistency with the
one real anchor B=0.49.

Run:  python scripts/window_rul.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import method_decision as M  # noqa: E402
from src.scoring import a_rul_score  # noqa: E402

FP = 600
STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}
W = 50          # window length (= test segment length)
TRAIN = [1, 2, 3, 4]


def hi_series(df, mu, sd):
    return M.hi(df, mu, sd)


def win_feats(hi, ot_end):
    """Shape features of a length-W HI window."""
    t = np.arange(len(hi)) * FP
    slope = np.polyfit(t, hi, 1)[0] * 3600.0          # HI per hour
    curv = np.polyfit(t, hi, 2)[0] if len(hi) > 2 else 0.0
    return [hi[-1], hi[0], hi[-1] - hi[0], hi.mean(), slope, curv * 1e9, ot_end]


def make_samples(bearings, his, ots):
    X, y, who = [], [], []
    for tr in bearings:
        h = his[tr]; ot = ots[tr]; n = len(h)
        for e in range(W, n + 1):                      # window = files [e-W, e)
            seg = h[e - W:e]
            X.append(win_feats(seg, ot[e - 1]))
            y.append(STOP[tr] - ((e - 1) * FP + 60))
            who.append(tr)
    return np.array(X), np.array(y, float), np.array(who)


def main():
    train = M.load(M.FO, TRAIN, "Train")
    test = M.load(M.FO / "test", M.TESTS, "")
    mu, sd = M.baseline(train)
    his = {t: hi_series(train[t], mu, sd) for t in TRAIN}
    # an extra raw degradation feature: max-channel OT_RMS per file
    ots = {t: train[t][[c for c in train[t].columns if c.endswith("OT_RMS")]].to_numpy(float).max(1)
           for t in TRAIN}

    X, y, who = make_samples(TRAIN, his, ots)
    print(f"windowed training samples: {len(X)} (from 4 trajectories)")

    # ---- LOO by bearing: predict held bearing's FIRST-50 window (test scenario) ----
    print("\n=== LOO (predict held bearing's first-50 window, /30) ===")
    scs = []
    for held in TRAIN:
        m = who != held
        rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=5, random_state=0)
        rf.fit(X[m], np.log1p(y[m]))
        seg = his[held][:W]
        feat = np.array([win_feats(seg, ots[held][W - 1])])
        pred = float(np.expm1(rf.predict(feat)[0]))
        act = STOP[held] - ((W - 1) * FP + 60)
        s = float(a_rul_score(act, max(600, pred)))
        scs.append(s)
        print(f"  Train{held}: pred={pred:>8.0f} actual={act:>6d} score={s:.3f}")
    print(f"  >> windowed-RF LOO mean = {np.mean(scs):.3f}  (const 0.561, isotonic 0.466)")

    # ---- fit on ALL training windows, predict the 6 validation bearings ----
    rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=5, random_state=0)
    rf.fit(X, np.log1p(y))
    print("\n=== validation predictions (windowed-RF) ===")
    B = {"Test1": 59351, "Test2": 51138, "Test3": 19936, "Test4": 12168, "Test5": 1800, "Test6": 1800}
    preds = {}
    print(f"{'bearing':>8} {'HI@50':>6} {'pred_RUL(s)':>12} {'(h)':>6}")
    for n in M.TESTS:
        h = hi_series(test[n], mu, sd)
        ot = test[n][[c for c in test[n].columns if c.endswith("OT_RMS")]].to_numpy(float).max(1)
        feat = np.array([win_feats(h[:W], ot[W - 1])])
        p = float(np.expm1(rf.predict(feat)[0]))
        preds[n] = max(600.0, p)
        print(f"{n:>8} {h[-1]:>6.2f} {preds[n]:>12.1f} {preds[n]/3600:>6.2f}")
    implied = float(np.mean([float(a_rul_score(preds[n], B[n])) for n in M.TESTS]))
    print(f"\nimplied B score = {implied:.3f}  (real 0.49, |gap|={abs(implied-0.49):.3f})")
    print("vector (s):", {n: round(preds[n], 1) for n in M.TESTS})


if __name__ == "__main__":
    main()
