"""Degradation-rate / first-passage-time RUL extrapolation for KSPHM-KIMM 2026.

Per task spec (analysis only). Follows the memo's RUL convention:
  total_life(N files) = (N-1)*600 + 60
  actual RUL at file d = (N-d)*600 + 60
Uses cached est features: outputs/ot_features/{est|test}/...

Method:
  1. HI = z-scored mean of selected features, rolling-median smoothed,
     optionally cummax (irreversible degradation).
  2. Robust degradation rate over last k files via Theil-Sen (median of pairwise
     slopes). Linear: HI ~ a + b t. Exponential: log(HI - base + eps) ~ a + b t.
  3. First-passage: project HI to threshold D = median EOL HI of the calibration
     bearings; RUL = time to reach D.
  4. Regularize: require b>0 else fall back to conservative constant (median of
     calibration bearings' actual RUL at the cut); clamp to [1800, max train life].

Also: LOO at cuts {40,50,60}. Wiener-FHT benchmarked separately (import wiener_rul).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.scoring import a_rul_score, error_pct  # noqa: E402

FEAT = ROOT / "outputs" / "ot_features"
FILE_PERIOD = 600
CHS = (0, 1, 2, 3)
TRAINS = (1, 2, 3, 4)
TESTS = (1, 2, 3, 4, 5, 6)

FEATSETS = {
    "transfer": ("Order_BandEnergy", "Spectral_Entropy"),
    "all6": ("OT_RMS", "OT_Kurtosis", "OT_CrestFactor",
             "Order_BandEnergy", "Spectral_Entropy", "Env_BandEnergy"),
    "energy": ("OT_RMS", "Order_BandEnergy", "Env_BandEnergy"),
}


def load_train():
    d = {}
    for tr in TRAINS:
        df = pd.read_csv(FEAT / "est" / f"Train{tr}.csv").sort_values(
            "File_Index").reset_index(drop=True)
        d[tr] = df
    return d


def load_test():
    d = {}
    for te in TESTS:
        df = pd.read_csv(FEAT / "test" / f"Test{te}.csv").sort_values(
            "File_Index").reset_index(drop=True)
        d[te] = df
    return d


def n_files(df):
    return len(df)


def total_life(df):
    return (n_files(df) - 1) * FILE_PERIOD + 60


def actual_rul_at(df, cut):
    # cut = number of files observed (1-indexed file count)
    return (n_files(df) - cut) * FILE_PERIOD + 60


def feat_cols(df, feats):
    return [f"Ch{c}_{f}" for c in CHS for f in feats if f"Ch{c}_{f}" in df.columns]


def build_hi(df, feats, mu, sd, smooth=5, cummax=True):
    X = df[feat_cols(df, feats)].to_numpy(float)
    z = ((X - mu) / sd).mean(1)
    s = pd.Series(z).rolling(smooth, min_periods=1).median()
    out = s.to_numpy()
    if cummax:
        out = pd.Series(out).cummax().to_numpy()
    return out


def baseline_stats(train, feats, healthy=0.15):
    pools = []
    for df in train.values():
        X = df[feat_cols(df, feats)].to_numpy(float)
        pools.append(X[: max(3, int(len(X) * healthy))])
    P = np.vstack(pools)
    return P.mean(0), P.std(0) + 1e-9


def theil_sen(x, y):
    """Median of pairwise slopes (robust)."""
    n = len(x)
    sl = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                sl.append((y[j] - y[i]) / dx)
    return float(np.median(sl)) if sl else 0.0


def rate_rul(hi_cut, D, k, mode, fallback, life_cap):
    """RUL via first-passage of a robust-slope projection.
    hi_cut: HI up to & including cut. mode: 'linear' | 'exp'.
    fallback: conservative RUL when not degrading. life_cap: clamp ceiling.
    """
    h_now = float(hi_cut[-1])
    w = min(k, len(hi_cut))
    if w < 3:
        return fallback
    y = hi_cut[-w:].astype(float)
    t = np.arange(w) * FILE_PERIOD  # local time, sec

    if mode == "linear":
        b = theil_sen(t, y)
        if b <= 1e-12:
            return fallback
        a = D - h_now
        if a <= 0:
            return 1800.0
        rul = a / b
    elif mode == "exp":
        base = min(y.min(), h_now) - 1e-6
        ly = np.log(np.maximum(y - base, 1e-9))
        b = theil_sen(t, ly)
        if b <= 1e-12:
            return fallback
        lD = np.log(max(D - base, 1e-9))
        l_now = np.log(max(h_now - base, 1e-9))
        if lD <= l_now:
            return 1800.0
        rul = (lD - l_now) / b
    else:
        raise ValueError(mode)
    return float(np.clip(rul, 1800.0, life_cap))


def loo_rate(train, feats, k, mode, cuts=(40, 50, 60), smooth=5, cummax=True,
             D_stat="median"):
    mu, sd = baseline_stats(train, feats)
    hi = {tr: build_hi(train[tr], feats, mu, sd, smooth, cummax) for tr in train}
    eol_hi = {tr: float(hi[tr][-1]) for tr in train}
    max_life = max(total_life(train[tr]) for tr in train)
    rows = []
    for held in train:
        others = [t for t in train if t != held]
        D = (float(np.median([eol_hi[t] for t in others])) if D_stat == "median"
             else float(np.percentile([eol_hi[t] for t in others], 25)))
        for cut in cuts:
            if cut >= n_files(train[held]):
                continue
            act = actual_rul_at(train[held], cut)
            fb = float(np.median([actual_rul_at(train[t], cut) for t in others]))
            life_cap = max_life
            rul = rate_rul(hi[held][:cut], D, k, mode, fb, life_cap)
            rows.append(dict(held=held, cut=cut, act=act, pred=rul,
                             er=float(error_pct(act, rul)),
                             score=float(a_rul_score(act, rul))))
    return pd.DataFrame(rows)


def predict_test(train, test, feats, k, mode, smooth=5, cummax=True,
                 D_stat="median"):
    """Calibrate D on ALL 4 training EOL HIs; predict the 6 test bearings at file 50."""
    mu, sd = baseline_stats(train, feats)
    hi_tr = {tr: build_hi(train[tr], feats, mu, sd, smooth, cummax) for tr in train}
    eol_hi = [float(hi_tr[tr][-1]) for tr in train]
    D = (float(np.median(eol_hi)) if D_stat == "median"
         else float(np.percentile(eol_hi, 25)))
    max_life = max(total_life(train[tr]) for tr in train)
    fb = float(np.median([actual_rul_at(train[tr], 50) for tr in train]))
    preds = {}
    for te in test:
        hi = build_hi(test[te], feats, mu, sd, smooth, cummax)
        preds[te] = rate_rul(hi, D, k, mode, fb, max_life)
    return preds, D


# ---------------- Wiener-FHT benchmark on the SAME index-cut LOO ----------------
def loo_wiener(train, feats, window, pct, cuts=(40, 50, 60), smooth=5,
               D_stat="median"):
    """Wiener first-passage (Inverse-Gaussian low-percentile), reusing the est
    features and the memo's index-cut + RUL convention so it is directly
    comparable to the rate model. Mirrors scripts/wiener_rul.py logic."""
    from scipy.stats import invgauss
    mu_b, sd_b = baseline_stats(train, feats)
    hi = {tr: build_hi(train[tr], feats, mu_b, sd_b, smooth, cummax=True)
          for tr in train}
    eol_hi = {tr: float(hi[tr][-1]) for tr in train}
    rows = []
    for held in train:
        others = [t for t in train if t != held]
        D = (float(np.median([eol_hi[t] for t in others])) if D_stat == "median"
             else float(np.percentile([eol_hi[t] for t in others], 25)))
        life_cap_full = float(np.percentile([total_life(train[t]) for t in others], 25))
        for cut in cuts:
            if cut >= n_files(train[held]):
                continue
            act = actual_rul_at(train[held], cut)
            t_cut = np.arange(cut) * FILE_PERIOD
            h = hi[held][:cut]
            elapsed = (cut - 1) * FILE_PERIOD
            cap = max(life_cap_full - elapsed, 1800.0)
            rul = _wiener_fpt(h, t_cut, D, window, pct, cap, invgauss)
            rows.append(dict(held=held, cut=cut, act=act, pred=rul,
                             er=float(error_pct(act, rul)),
                             score=float(a_rul_score(act, rul))))
    return pd.DataFrame(rows)


def _wiener_fpt(hi_cut, t_cut, D, window, pct, cap, invgauss, floor=1800.0):
    cap = max(cap, floor)
    h_now = hi_cut[-1]
    a = D - h_now
    if a <= 0:
        return floor
    w = min(window, len(hi_cut))
    if w < 3:
        return cap
    y = hi_cut[-w:]; t = t_cut[-w:]
    tm = t - t.mean(); denom = (tm ** 2).sum()
    mu = float((tm * (y - y.mean())).sum() / denom) if denom > 0 else 0.0
    dy = np.diff(y); dt = np.diff(t)
    e = dy - mu * dt
    sigma2 = float(np.mean(e ** 2) / np.mean(dt)) if len(e) else 0.0
    if mu <= 1e-12:
        return cap
    mean_T = a / mu
    if sigma2 <= 1e-18:
        rul = mean_T
    else:
        lam = a ** 2 / sigma2
        rul = float(invgauss.ppf(pct, mu=mean_T / lam, scale=lam))
        if not np.isfinite(rul):
            rul = mean_T
    return float(min(max(rul, floor), cap))


def predict_test_wiener(train, test, feats, window, pct, smooth=5, D_stat="p25"):
    from scipy.stats import invgauss
    mu_b, sd_b = baseline_stats(train, feats)
    hi_tr = {tr: build_hi(train[tr], feats, mu_b, sd_b, smooth, cummax=True)
             for tr in train}
    eol_hi = [float(hi_tr[tr][-1]) for tr in train]
    D = (float(np.median(eol_hi)) if D_stat == "median"
         else float(np.percentile(eol_hi, 25)))
    life_cap_full = float(np.percentile([total_life(train[t]) for t in train], 25))
    elapsed = (50 - 1) * FILE_PERIOD
    cap = max(life_cap_full - elapsed, 1800.0)
    preds = {}
    for te in test:
        hi = build_hi(test[te], feats, mu_b, sd_b, smooth, cummax=True)
        t_cut = np.arange(len(hi)) * FILE_PERIOD
        preds[te] = _wiener_fpt(hi, t_cut, D, window, pct, cap, invgauss)
    return preds, D


def main():
    train = load_train()
    test = load_test()

    print("=== RUL convention (memo) ===")
    for tr in TRAINS:
        print(f"  Train{tr}: N={n_files(train[tr])}, total_life={total_life(train[tr])}, "
              f"actRUL@50={actual_rul_at(train[tr],50)}")

    print("\n=== RATE MODEL: LOO sweep (feats x k x mode), cuts 40/50/60 ===")
    print(f"{'feats':>9} {'k':>3} {'mode':>7} {'cut50':>7} {'avg':>7}")
    results = []
    for fs in ("transfer", "all6", "energy"):
        for k in (10, 15, 20):
            for mode in ("linear", "exp"):
                df = loo_rate(train, FEATSETS[fs], k, mode)
                cut50 = df[df.cut == 50].score.mean()
                avg = df.score.mean()
                results.append((fs, k, mode, cut50, avg, df))
                print(f"{fs:>9} {k:>3} {mode:>7} {cut50:>7.4f} {avg:>7.4f}")

    # pick best by cut50 then avg
    best = max(results, key=lambda r: (r[3], r[4]))
    r_fs, r_k, r_mode, cut50, avg, bdf = best
    print(f"\n>>> BEST RATE: feats={r_fs} k={r_k} mode={r_mode}  cut50={cut50:.4f} avg={avg:.4f}")
    print("    per-bearing LOO detail (best config):")
    print(bdf.to_string(index=False))

    print("\n=== WIENER-FHT: LOO sweep on same index cuts ===")
    print(f"{'feats':>9} {'win':>4} {'pct':>5} {'cut50':>7} {'avg':>7}")
    wres = []
    for fs in ("transfer", "all6", "energy"):
        for window in (6, 10, 15):
            for pct in (0.2, 0.3, 0.5):
                df = loo_wiener(train, FEATSETS[fs], window, pct)
                cut50 = df[df.cut == 50].score.mean()
                avg = df.score.mean()
                wres.append((fs, window, pct, cut50, avg, df))
                print(f"{fs:>9} {window:>4} {pct:>5} {cut50:>7.4f} {avg:>7.4f}")
    wbest = max(wres, key=lambda r: (r[3], r[4]))
    wfs, wwin, wpct, wcut50, wavg, wbdf = wbest
    print(f"\n>>> BEST WIENER: feats={wfs} win={wwin} pct={wpct} "
          f"cut50={wcut50:.4f} avg={wavg:.4f}")
    print(wbdf.to_string(index=False))

    print("\n=== TEST PREDICTIONS @ file 50 ===")
    preds_r, Dr = predict_test(train, test, FEATSETS[r_fs], r_k, r_mode)
    print(f"RATE (feats={r_fs} k={r_k} mode={r_mode}, D={Dr:.3f}):")
    for te in TESTS:
        print(f"  Val{te} = {int(round(preds_r[te]))}")
    preds_w, Dw = predict_test_wiener(train, test, FEATSETS[wfs], wwin, wpct)
    print(f"WIENER (feats={wfs} win={wwin} pct={wpct}, D={Dw:.3f}):")
    for te in TESTS:
        print(f"  Val{te} = {int(round(preds_w[te]))}")

    print(f"\nVal2 raw (rate)   = {int(round(preds_r[2]))}  (vs ground-truth ~72906)")
    print(f"Val2 raw (wiener) = {int(round(preds_w[2]))}")

    # stability: report HI slope sign & raw extrapolation per test bearing (rate)
    print("\n=== STABILITY: rate-model raw slope diagnostics per test bearing ===")
    mu_b, sd_b = baseline_stats(train, FEATSETS[r_fs])
    for te in TESTS:
        hi = build_hi(test[te], FEATSETS[r_fs], mu_b, sd_b)
        w = min(r_k, len(hi))
        t = np.arange(w) * FILE_PERIOD
        b = theil_sen(t, hi[-w:])
        print(f"  Val{te}: HI_now={hi[-1]:8.3f}  slope(/s)={b:.3e}  "
              f"clamped={'yes' if (b>1e-12 and not (1800<= (Dr-hi[-1])/b <= max(total_life(train[tr]) for tr in TRAINS))) else 'no' if b>1e-12 else 'FLAT->fallback'}")

    # save outputs
    outdir = ROOT / "outputs" / "scratch" / "rate"
    outdir.mkdir(parents=True, exist_ok=True)
    bdf.to_csv(outdir / "loo_rate_best.csv", index=False)
    wbdf.to_csv(outdir / "loo_wiener_best.csv", index=False)
    pd.DataFrame([dict(bearing=f"Val{te}", rate=int(round(preds_r[te])),
                       wiener=int(round(preds_w[te]))) for te in TESTS]).to_csv(
        outdir / "test_predictions.csv", index=False)
    pd.DataFrame([dict(feats=r[0], k=r[1], mode=r[2], cut50=r[3], avg=r[4])
                  for r in results]).to_csv(outdir / "rate_sweep.csv", index=False)
    pd.DataFrame([dict(feats=r[0], window=r[1], pct=r[2], cut50=r[3], avg=r[4])
                  for r in wres]).to_csv(outdir / "wiener_sweep.csv", index=False)
    print(f"\nsaved -> {outdir}")


if __name__ == "__main__":
    main()
