"""Head-to-head model benchmark on the test-faithful evaluator (cut 40/50/60 LOO,
official /30 A_RUL). No dogma: each candidate predicts the held-out bearing's RUL
from only its first `cut` files; we score and rank. B's real score (0.49) anchors
the harness's absolute level (LOO is known to be biased).

Candidates:
  HI+const     : ignore HI level, predict median life of other bearings  (current best)
  HI+isotonic  : monotone RUL=g(HI) fit on training cloud
  lifefrac     : HI -> life fraction -> RUL (~ what Option B did, real 0.49)
  ElasticNet   : regularized linear regression on ALL features -> log1p(RUL)
  RandomForest : shallow RF on ALL features -> log1p(RUL)
  Weibull-life : survival-style, RUL from Weibull fit of lifetimes (no features)

Run:  python scripts/benchmark_models.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from scipy.stats import weibull_min

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import method_decision as M  # noqa: E402
from src.scoring import a_rul_score  # noqa: E402

FP = 600
STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}
CUTS = (40, 50, 60)
TRAIN = [1, 2, 3, 4]


def feat_cols(df):
    return [c for c in df.columns if c.startswith("Ch")]


def build(train):
    """Per-file (features, HI, RUL) for every training file."""
    mu, sd = M.baseline(train)
    data = {}
    for tr in train:
        df = train[tr]
        t = (df["File_Index"].to_numpy() - 1) * FP
        data[tr] = dict(X=df[feat_cols(df)].to_numpy(float),
                        hi=M.hi(df, mu, sd),
                        rul=(STOP[tr] - (t + 60)).astype(float),
                        n=len(df))
    return data, mu, sd


def predict(model, held, cut, data, train):
    others = [t for t in TRAIN if t != held]
    # assemble training rows from others (all their files)
    Xtr = np.vstack([data[t]["X"] for t in others])
    rultr = np.concatenate([data[t]["rul"] for t in others])
    hitr = np.concatenate([data[t]["hi"] for t in others])
    xh = data[held]["X"][cut - 1]
    hih = data[held]["hi"][cut - 1]
    other_rul_at = [STOP[t] - ((cut - 1) * FP + 60) for t in others]

    if model == "HI+const":
        return float(np.median(other_rul_at))
    if model == "Weibull-life":
        lives = np.array([STOP[t] for t in others], float)
        c, loc, scale = weibull_min.fit(lives, floc=0)
        elapsed = (cut - 1) * FP + 60
        # expected remaining life given survival to `elapsed`
        xs = np.linspace(elapsed, lives.max() * 3, 4000)
        sf = weibull_min.sf(xs, c, loc, scale)
        if sf[0] <= 1e-9:
            return float(np.median(lives) - elapsed)
        exp_life = elapsed + np.trapz(sf, xs) / sf[0]
        return float(max(600, exp_life - elapsed))
    if model == "HI+isotonic":
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(hitr, rultr)
        return float(iso.predict([hih])[0])
    if model == "lifefrac":
        other_life = [STOP[t] for t in others]
        fracs = []
        for t in others:
            ht = data[t]["hi"]; idx = np.argmax(ht >= hih)
            fracs.append(idx / len(ht) if ht[idx] >= hih else 1.0)
        return float(max(600, (1 - np.median(fracs)) * np.median(other_life)))
    if model in ("ElasticNet", "RandomForest"):
        ytr = np.log1p(rultr)
        if model == "ElasticNet":
            reg = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, max_iter=5000))
        else:
            reg = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=0)
        reg.fit(Xtr, ytr)
        return float(np.expm1(reg.predict(xh.reshape(1, -1))[0]))
    raise ValueError(model)


def main():
    train = M.load(M.FO, TRAIN, "Train")
    test = M.load(M.FO / "test", M.TESTS, "")
    data, mu, sd = build(train)
    models = ["HI+const", "HI+isotonic", "lifefrac", "Weibull-life", "ElasticNet", "RandomForest"]

    print("=== LOO benchmark (cuts 40/50/60, official /30) ===")
    print(f"{'model':>14} {'LOO A_RUL':>10}   per-cut50 scores (T1..T4)")
    loo = {}
    for m in models:
        pts = []
        per50 = []
        for cut in CUTS:
            for held in TRAIN:
                act = STOP[held] - ((cut - 1) * FP + 60)
                pred = predict(m, held, cut, data, train)
                s = float(a_rul_score(act, max(600, pred)))
                pts.append(s)
                if cut == 50:
                    per50.append(s)
        loo[m] = np.mean(pts)
        print(f"{m:>14} {np.mean(pts):>10.3f}   " + " ".join(f"{s:.2f}" for s in per50))

    # test predictions per model + CONSISTENCY CHECK against the one real anchor (B=0.49):
    # if a model's predicted RULs were the truth, B's vector would score `implied_B`.
    # The closer implied_B is to the real 0.49, the more consistent the model is with reality.
    B = {"Test1": 59351, "Test2": 51138, "Test3": 19936, "Test4": 12168, "Test5": 1800, "Test6": 1800}
    print("\n=== test predictions per model (cut=50, sec) + consistency with B=0.49 ===")
    implied = {}
    for m in models:
        preds = {}
        for n in M.TESTS:
            # build a pseudo 'held' = test bearing using all 4 train as others
            Xtr = np.vstack([data[t]["X"] for t in TRAIN])
            rultr = np.concatenate([data[t]["rul"] for t in TRAIN])
            hitr = np.concatenate([data[t]["hi"] for t in TRAIN])
            xh = test[n][feat_cols(test[n])].to_numpy(float)[49]
            hih = M.hi(test[n], mu, sd)[49]
            if m == "HI+const":
                p = float(np.median([STOP[t] - (49 * FP + 60) for t in TRAIN]))
            elif m == "Weibull-life":
                lives = np.array([STOP[t] for t in TRAIN], float)
                c, loc, scale = weibull_min.fit(lives, floc=0)
                elapsed = 49 * FP + 60
                xs = np.linspace(elapsed, lives.max() * 3, 4000); sf = weibull_min.sf(xs, c, loc, scale)
                p = float(elapsed + np.trapz(sf, xs) / max(sf[0], 1e-9) - elapsed)
            elif m == "HI+isotonic":
                p = float(IsotonicRegression(increasing=False, out_of_bounds="clip").fit(hitr, rultr).predict([hih])[0])
            elif m == "lifefrac":
                fr = []
                for t in TRAIN:
                    ht = data[t]["hi"]; idx = np.argmax(ht >= hih); fr.append(idx / len(ht) if ht[idx] >= hih else 1.0)
                p = float(max(600, (1 - np.median(fr)) * np.median([STOP[t] for t in TRAIN])))
            else:
                ytr = np.log1p(rultr)
                reg = (make_pipeline(StandardScaler(), ElasticNetCV(cv=3, max_iter=5000)) if m == "ElasticNet"
                       else RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=0))
                reg.fit(Xtr, ytr); p = float(np.expm1(reg.predict(xh.reshape(1, -1))[0]))
            preds[n] = int(round(max(600, p)))
        implied[m] = float(np.mean([float(a_rul_score(preds[n], B[n])) for n in M.TESTS]))
        print(f"  {m:>14}: " + " ".join(f"{preds[n]:>6d}" for n in M.TESTS) +
              f"   | implied B={implied[m]:.3f} (real 0.49, |gap|={abs(implied[m]-0.49):.3f})")

    print("\n=== summary: LOO quality vs consistency-with-reality (B=0.49) ===")
    for m in sorted(models, key=lambda x: abs(implied[x] - 0.49)):
        print(f"  {m:>14}: LOO={loo[m]:.3f}  implied_B={implied[m]:.3f}  |gap to 0.49|={abs(implied[m]-0.49):.3f}")


if __name__ == "__main__":
    main()
