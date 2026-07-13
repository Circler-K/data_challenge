"""KSPHM-KIMM 2026 — Survival / total-life RUL prediction (analysis only).

Method:
    RUL = total_life - elapsed,  elapsed@file50 = 49*600 = 29400 s.
  Predict TOTAL life, subtract elapsed.

  (1) Pure-distribution baseline: fit Weibull & lognormal to the 4 training
      total-lives, compute censoring-aware conditional expected residual life
      E[L - t | L > t] at t=29400. Same prediction for all bearings.
  (2) Feature-modulated: regress total_life on a first-50-file degradation
      descriptor (LOO), apply to test descriptors.
  (3) Compare on leave-one-out A_RUL (cut each training bearing at d in {40,50,60},
      calibrate on the OTHER 3, score vs known actual RUL).

Outputs to outputs/scratch/survival/. Does NOT touch submission files.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy import stats, integrate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.scoring import a_rul_score, final_score  # noqa

FILE_PERIOD = 600.0
ELAPSED_50 = 49 * FILE_PERIOD          # 29400 s elapsed at file 50
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "scratch", "survival")
os.makedirs(OUTDIR, exist_ok=True)

EST = os.path.join(os.path.dirname(__file__), "..", "outputs", "ot_features", "est")
TST = os.path.join(os.path.dirname(__file__), "..", "outputs", "ot_features", "test")

# Training file counts (number of files N). total_life = (N-1)*600 + 60
TRAIN_N = {1: 126, 2: 114, 3: 89, 4: 137}


def total_life(N):
    return (N - 1) * FILE_PERIOD + 60.0


def actual_rul_at_file(N, d):
    # d = file index (1-based). elapsed = (d-1)*600. RUL = total - elapsed
    return total_life(N) - (d - 1) * FILE_PERIOD


# ---------------------------------------------------------------- descriptors
CHS = [0, 1, 2, 3]


def load_train(i):
    return pd.read_csv(os.path.join(EST, f"Train{i}.csv"))


def load_test(i):
    return pd.read_csv(os.path.join(TST, f"Test{i}.csv"))


def health_index(df):
    """Simple multi-channel HI: mean of z-normalised RMS across 4 channels.
    Normalisation uses the bearing's own first-10-file baseline (per-bearing,
    no cross-bearing leakage)."""
    rms = np.column_stack([df[f"Ch{c}_OT_RMS"].values for c in CHS])
    base_mu = rms[:10].mean(axis=0)
    base_sd = rms[:10].std(axis=0) + 1e-9
    z = (rms - base_mu) / base_sd
    return z.mean(axis=1)


def descriptor_first_d(df, d):
    """Degradation descriptors from files 1..d (rows 0..d-1)."""
    sub = df.iloc[:d]
    hi = health_index(df)[:d]
    x_idx = np.arange(d)
    slope = np.polyfit(x_idx, hi, 1)[0] if d > 1 else 0.0
    return {
        "hi_level": float(hi[-1]),
        "hi_slope": float(slope),
        "mean_order": float(np.mean([sub[f"Ch{c}_Order_BandEnergy"].mean() for c in CHS])),
        "mean_entropy": float(np.mean([sub[f"Ch{c}_Spectral_Entropy"].mean() for c in CHS])),
        "mean_rms": float(np.mean([sub[f"Ch{c}_OT_RMS"].iloc[:d].mean() for c in CHS])),
    }


# ---------------------------------------------------------------- distributions
def fit_weibull(lives):
    # 2-param Weibull (floc=0)
    c, loc, scale = stats.weibull_min.fit(lives, floc=0)
    return ("weibull", (c, loc, scale))


def fit_lognorm(lives):
    s, loc, scale = stats.lognorm.fit(lives, floc=0)
    return ("lognorm", (s, loc, scale))


def cond_expected_residual(dist_name, params, t):
    """E[L - t | L > t] = integral_t^inf S(x) dx / S(t)."""
    if dist_name == "weibull":
        d = stats.weibull_min(*params)
    else:
        d = stats.lognorm(*params)
    St = d.sf(t)
    if St <= 1e-12:
        # survived past mass — fall back to a long horizon residual
        St = 1e-12
    upper = d.ppf(0.99999)
    upper = max(upper, t * 5)
    val, _ = integrate.quad(lambda x: d.sf(x), t, upper, limit=200)
    return val / St


# ---------------------------------------------------------------- feature model
def fit_feature_model(train_ids, descr_key, d):
    """LOO-safe: caller passes the subset of training ids to calibrate on.
    Regress log(total_life) on a single descriptor (robust for n=3-4)."""
    xs, ys = [], []
    for i in train_ids:
        df = load_train(i)
        x = descriptor_first_d(df, d)[descr_key]
        xs.append(x)
        ys.append(np.log(total_life(TRAIN_N[i])))
    xs = np.array(xs); ys = np.array(ys)
    # linear fit log(total) = a*x + b
    a, b = np.polyfit(xs, ys, 1)
    return (a, b)


def predict_feature(model, x):
    a, b = model
    return float(np.exp(a * x + b))


# --- robust linear feature model with extrapolation clipping -----------------
def fit_feature_model_lin(train_ids, descr_key, d):
    """Linear (not log) total_life = a*x + b. Returns (a,b, xmin,xmax, life_lo,life_hi)
    so predictions can be clipped to the training-observed lifetime envelope
    (guards against exponential blow-up on out-of-range test descriptors)."""
    xs, ys = [], []
    for i in train_ids:
        df = load_train(i)
        xs.append(descriptor_first_d(df, d)[descr_key])
        ys.append(total_life(TRAIN_N[i]))
    xs = np.array(xs); ys = np.array(ys)
    a, b = np.polyfit(xs, ys, 1)
    # envelope: clip total-life to [min observed, max observed * 1.15] (allow
    # modest right-censored extension since validation bearings can outlive training)
    return (a, b, xs.min(), xs.max(), ys.min(), ys.max())


def predict_feature_lin(model, x, clip=True):
    a, b, xmin, xmax, life_lo, life_hi = model
    if clip:
        x = float(np.clip(x, xmin, xmax))   # no extrapolation beyond observed descriptor range
    tot = a * x + b
    if clip:
        tot = float(np.clip(tot, life_lo, life_hi * 1.20))
    return float(tot)


# ---------------------------------------------------------------- LOO evaluation
def run_loo(descr_key="hi_level", cuts=(40, 50, 60)):
    ids = [1, 2, 3, 4]
    rows = []
    for d in cuts:
        for held in ids:
            others = [i for i in ids if i != held]
            lives_other = np.array([total_life(TRAIN_N[i]) for i in others])
            elapsed = (d - 1) * FILE_PERIOD
            actual = actual_rul_at_file(TRAIN_N[held], d)

            # distribution baselines (calibrated on others only)
            wb = fit_weibull(lives_other)
            ln = fit_lognorm(lives_other)
            pred_wb = cond_expected_residual(*wb, t=elapsed)
            pred_ln = cond_expected_residual(*ln, t=elapsed)

            # feature model (calibrated on others only)
            fm = fit_feature_model(others, descr_key, d)
            df_held = load_train(held)
            xheld = descriptor_first_d(df_held, d)[descr_key]
            tot_pred = predict_feature(fm, xheld)
            pred_feat = tot_pred - elapsed

            # robust clipped-linear feature model
            fml = fit_feature_model_lin(others, descr_key, d)
            tot_lin = predict_feature_lin(fml, xheld)
            pred_lin = tot_lin - elapsed

            rows.append(dict(d=d, held=held, actual=actual,
                             pred_wb=pred_wb, pred_ln=pred_ln,
                             pred_feat=pred_feat, pred_lin=pred_lin,
                             s_wb=float(a_rul_score(actual, pred_wb)),
                             s_ln=float(a_rul_score(actual, pred_ln)),
                             s_feat=float(a_rul_score(actual, max(pred_feat, 1.0))),
                             s_lin=float(a_rul_score(actual, max(pred_lin, 1.0)))))
    return pd.DataFrame(rows)


def summarize(loo):
    out = {}
    for col, label in [("s_wb", "weibull"), ("s_ln", "lognorm"),
                       ("s_feat", "feat_logexp"), ("s_lin", "feat_linclip")]:
        cut50 = loo[loo.d == 50][col].mean()
        allc = loo[col].mean()
        out[label] = (cut50, allc)
    return out


# ---------------------------------------------------------------- test predictions
def predict_tests(descr_key="hi_level"):
    ids = [1, 2, 3, 4]
    lives_all = np.array([total_life(TRAIN_N[i]) for i in ids])
    wb = fit_weibull(lives_all)
    ln = fit_lognorm(lives_all)
    # distribution baseline at t=29400 (same for all)
    base_wb = cond_expected_residual(*wb, t=ELAPSED_50)
    base_ln = cond_expected_residual(*ln, t=ELAPSED_50)

    fm = fit_feature_model(ids, descr_key, 50)
    fml = fit_feature_model_lin(ids, descr_key, 50)

    rows = []
    for ti in range(1, 7):
        dft = load_test(ti)
        x = descriptor_first_d(dft, 50)[descr_key]
        tot = predict_feature(fm, x)
        tot_lin = predict_feature_lin(fml, x)
        pred_feat = tot - ELAPSED_50
        pred_lin = tot_lin - ELAPSED_50
        # final blend: clipped-linear feature + distribution baseline, averaged,
        # then take the conservative (lower) leaning via 0.6 feat / 0.4 dist
        blend = 0.6 * pred_lin + 0.4 * base_ln
        rows.append(dict(test=ti, descr=x,
                         total_life_pred=tot,
                         total_life_lin=tot_lin,
                         rul_feat=pred_feat,
                         rul_lin=pred_lin,
                         rul_blend=blend,
                         rul_wb=base_wb,
                         rul_ln=base_ln))
    return pd.DataFrame(rows), (base_wb, base_ln), (wb, ln, lives_all)


if __name__ == "__main__":
    print("=== Training total-lives & RUL@50 ===")
    for i in [1, 2, 3, 4]:
        print(f"Train{i}: N={TRAIN_N[i]}  total_life={total_life(TRAIN_N[i]):.0f}  "
              f"RUL@file50={actual_rul_at_file(TRAIN_N[i],50):.0f}")

    # try several descriptors, report LOO
    print("\n=== LOO A_RUL by descriptor (feature model) ===")
    best = None
    for key in ["hi_level", "hi_slope", "mean_order", "mean_entropy", "mean_rms"]:
        loo = run_loo(descr_key=key)
        s = summarize(loo)
        print(f"\n[{key}]")
        for label, (c50, ca) in s.items():
            print(f"   {label:8s} cut50={c50:.4f}  avg(40,50,60)={ca:.4f}")
        # rank by the robust clipped-linear feature model's cut50
        if best is None or s["feat_linclip"][0] > best[1]:
            best = (key, s["feat_linclip"][0])
    print(f"\nBest feature descriptor by LOO cut50 (linclip): {best[0]} ({best[1]:.4f})")

    # full LOO detail for best descriptor and also distribution
    loo = run_loo(descr_key=best[0])
    loo.to_csv(os.path.join(OUTDIR, "loo_detail.csv"), index=False)
    print("\n=== LOO detail (best descriptor) ===")
    print(loo.to_string(index=False))

    # Test predictions
    print("\n=== TEST predictions (descriptor = %s) ===" % best[0])
    preds, (base_wb, base_ln), (wb, ln, lives_all) = predict_tests(descr_key=best[0])
    print("\nWeibull fit (c,loc,scale):", wb[1])
    print("Lognorm fit (s,loc,scale):", ln[1])
    print(f"Pure-distribution conditional residual @t=29400: weibull={base_wb:.0f}  lognorm={base_ln:.0f}")
    print("\n", preds.to_string(index=False))
    preds.to_csv(os.path.join(OUTDIR, "test_predictions.csv"), index=False)

    print("\n=== Final 6-bearing predictions (exact integer seconds) ===")
    print("V  : logexp-feat | linclip-feat | lognorm-base | BLEND(0.6lin+0.4dist)")
    for _, r in preds.iterrows():
        print(f"  V{int(r.test)}: {int(round(r.rul_feat)):11d} | {int(round(r.rul_lin)):11d} | "
              f"{int(round(r.rul_ln)):11d} | {int(round(r.rul_blend)):11d}")

    # V2 ground-truth comparison
    v2 = preds[preds.test == 2].iloc[0]
    print("\n=== Validation2 vs ground-truth ~72906 (band 69219-77276) ===")
    print(f"  logexp-feat  : {int(round(v2.rul_feat))}")
    print(f"  linclip-feat : {int(round(v2.rul_lin))}")
    print(f"  lognorm-base : {int(round(v2.rul_ln))}")
    print(f"  blend        : {int(round(v2.rul_blend))}")
    print(f"  hi_level-feat (logexp) for reference recomputed below")

    # also report hi_level-based V2 (the +0.87 confound descriptor)
    fm_hi = fit_feature_model([1,2,3,4], "hi_level", 50)
    x_hi = descriptor_first_d(load_test(2), 50)["hi_level"]
    v2_hi = predict_feature(fm_hi, x_hi) - ELAPSED_50
    print(f"  hi_level-feat: total={predict_feature(fm_hi,x_hi):.0f}  RUL={int(round(v2_hi))}")
