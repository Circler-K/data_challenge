"""Trajectory similarity matching for KSPHM-KIMM 2026 bearing RUL.

Analysis only. Does NOT touch submissions or existing scripts.
Outputs -> outputs/scratch/similarity/.
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring import a_rul_score, final_score

PERIOD = 600.0
EST = "outputs/ot_features/est"
TST = "outputs/ot_features/test"
OUT = "outputs/scratch/similarity"
os.makedirs(OUT, exist_ok=True)

TRAIN_IDS = [1, 2, 3, 4]
TRAIN_N = {1: 126, 2: 114, 3: 89, 4: 137}  # full lifetimes
TEST_IDS = [1, 2, 3, 4, 5, 6]

FEATS6 = ["OT_RMS", "OT_Kurtosis", "OT_CrestFactor",
          "Order_BandEnergy", "Spectral_Entropy", "Env_BandEnergy"]
FEATS_TRANSFER = ["Order_BandEnergy", "Spectral_Entropy"]
CHANS = [0, 1, 2, 3]


def life_total(n):       return (n - 1) * PERIOD + 60.0
def rul_at(n, d):        return (n - d) * PERIOD + 60.0   # d = file index (1-based)


def load_train(tid):
    return pd.read_csv(f"{EST}/Train{tid}.csv")

def load_test(tid):
    return pd.read_csv(f"{TST}/Test{tid}.csv")


def channel_aggregate(df, feat, mode="mean"):
    """Return per-file aggregated value for a single feature across channels."""
    cols = [f"Ch{c}_{feat}" for c in CHANS]
    sub = df[cols].values
    if mode == "max":
        return sub.max(axis=1)
    return sub.mean(axis=1)


def rolling_median(x, w=5):
    x = np.asarray(x, float)
    n = len(x)
    out = np.empty(n)
    h = w // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[i] = np.median(x[lo:hi])
    return out


def build_hi(df, feats, chan_mode="mean", norm=None, smooth=5):
    """Build a 1-D HI series of length len(df).
    norm: dict feat->(mean,std) from a fixed reference (for z-scoring), or None.
    Returns hi series and the per-feat (mean,std) used (computed on df if norm None)."""
    series = {}
    stats = {}
    for f in feats:
        agg = channel_aggregate(df, f, chan_mode)
        if norm is not None and f in norm:
            mu, sd = norm[f]
        else:
            mu, sd = np.mean(agg), np.std(agg) + 1e-9
        stats[f] = (mu, sd)
        series[f] = (agg - mu) / sd
    hi = np.mean(np.column_stack([series[f] for f in feats]), axis=1)
    if smooth and smooth > 1:
        hi = rolling_median(hi, smooth)
    return hi, stats


def build_hi_pca(df, feats, chan_mode="mean", pca_axes=None, smooth=5):
    """First principal component HI. If pca_axes given (mean,components) reuse it."""
    mats = [channel_aggregate(df, f, chan_mode) for f in feats]
    X = np.column_stack(mats)  # (T, n_feat)
    if pca_axes is None:
        mu = X.mean(axis=0)
        Xc = X - mu
        # std-normalize columns first
        sd = Xc.std(axis=0) + 1e-9
        Xc = Xc / sd
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comp = Vt[0]
        pca_axes = (mu, sd, comp)
    else:
        mu, sd, comp = pca_axes
    Xc = (X - mu) / sd
    hi = Xc @ comp
    if smooth and smooth > 1:
        hi = rolling_median(hi, smooth)
    return hi, pca_axes


# ---------------- distances on aligned length-L windows ----------------

def d_euclid(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def d_corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def dtw_dist(a, b, window=10):
    """Simple DP DTW with Sakoe-Chiba band."""
    n, m = len(a), len(b)
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    w = max(window, abs(n - m))
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - w), min(m, i + w)
        for j in range(jlo, jhi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def slope(x):
    t = np.arange(len(x))
    return float(np.polyfit(t, x, 1)[0])


# ---------------- prediction engine ----------------

def predict_one(test_hi_50, ref_lib, method, dist, temp=0.5):
    """ref_lib: list of (tid, ref_hi_50, rul50). Returns predicted RUL (float)
    and diagnostic dict."""
    cands = []
    for tid, ref_hi, rul50 in ref_lib:
        a, b = test_hi_50, ref_hi
        if dist == "euclid":
            d = d_euclid(a, b); sim = -d
        elif dist == "corr":
            c = d_corr(a, b); sim = c; d = 1 - c
        elif dist == "dtw":
            d = dtw_dist(a, b); sim = -d
        elif dist == "slope":  # match by trajectory slope only
            d = abs(slope(a) - slope(b)); sim = -d
        cands.append((tid, d, sim, rul50))
    cands.sort(key=lambda r: r[1])
    if method == "nn":
        pred = cands[0][3]
    elif method == "softmax":
        sims = np.array([c[2] for c in cands], float)
        ruls = np.array([c[3] for c in cands], float)
        # scale sims to comparable range
        s = sims / (np.std(sims) + 1e-9)
        w = np.exp(s / temp)
        w /= w.sum()
        pred = float(np.dot(w, ruls))
    return pred, cands


def make_ref_lib(hi_kind, feats, chan_mode, exclude=None, fit_norm_on="self",
                 cut=50, smooth=5):
    """Build reference library of training bearings cut at `cut` files.
    Returns list of (tid, hi_cut, rul_at_cut) and the norm/pca info per tid."""
    lib = []
    for tid in TRAIN_IDS:
        if exclude is not None and tid == exclude:
            continue
        df = load_train(tid)
        df = df[df.File_Index <= cut].reset_index(drop=True)
        if hi_kind == "pca":
            hi, _ = build_hi_pca(df, feats, chan_mode, None, smooth)
        else:
            hi, _ = build_hi(df, feats, chan_mode, None, smooth)
        lib.append((tid, hi, rul_at(TRAIN_N[tid], cut)))
    return lib


def test_hi_series(tid, hi_kind, feats, chan_mode, smooth=5):
    df = load_test(tid)
    if hi_kind == "pca":
        hi, _ = build_hi_pca(df, feats, chan_mode, None, smooth)
    else:
        hi, _ = build_hi(df, feats, chan_mode, None, smooth)
    return hi


def train_hi_cut(tid, cut, hi_kind, feats, chan_mode, smooth=5):
    df = load_train(tid)
    df = df[df.File_Index <= cut].reset_index(drop=True)
    if hi_kind == "pca":
        hi, _ = build_hi_pca(df, feats, chan_mode, None, smooth)
    else:
        hi, _ = build_hi(df, feats, chan_mode, None, smooth)
    return hi


# ---------------- LOO evaluation across config grid ----------------

def loo_score(hi_kind, feats, chan_mode, method, dist, cuts=(40, 50, 60),
              smooth=5, temp=0.5, censor_extrap=False):
    """For each cut, leave-one-bearing-out: predict the held-out bearing using
    the other 3, score against true RUL@cut. Returns dict cut->score, and avg."""
    per_cut = {}
    for cut in cuts:
        acts, preds = [], []
        for held in TRAIN_IDS:
            lib = []
            for tid in TRAIN_IDS:
                if tid == held:
                    continue
                hi = train_hi_cut(tid, cut, hi_kind, feats, chan_mode, smooth)
                lib.append((tid, hi, rul_at(TRAIN_N[tid], cut)))
            test_hi = train_hi_cut(held, cut, hi_kind, feats, chan_mode, smooth)
            pred, cands = predict_one(test_hi, lib, method, dist, temp)
            if censor_extrap:
                pred = apply_censor(test_hi, lib, pred, cut)
            acts.append(rul_at(TRAIN_N[held], cut))
            preds.append(pred)
        per_cut[cut] = final_score(np.array(acts), np.array(preds))
    avg = float(np.mean(list(per_cut.values())))
    return per_cut, avg


def apply_censor(test_hi, lib, pred, cut):
    """If test bearing's HI level is lower (healthier) than all refs, allow
    extrapolation above max ref RUL@cut proportional to how much flatter it is."""
    test_lvl = np.mean(test_hi[-10:])
    ref_lvls = [np.mean(h[-10:]) for _, h, _ in lib]
    max_rul = max(r for _, _, r in lib)
    if test_lvl < min(ref_lvls):
        # extrapolate: scale up modestly (cap at +50%)
        gap = (min(ref_lvls) - test_lvl) / (np.std(ref_lvls) + 1e-9)
        factor = 1.0 + min(0.5, 0.15 * gap)
        return max(pred, max_rul * factor)
    return pred


def grid_search():
    hi_kinds = ["zall", "ztransfer", "pca"]
    feat_map = {"zall": FEATS6, "ztransfer": FEATS_TRANSFER, "pca": FEATS6}
    chan_modes = ["mean", "max"]
    methods = ["nn", "softmax"]
    dists = ["euclid", "corr", "dtw", "slope"]
    rows = []
    for hk, cm, mt, ds in itertools.product(hi_kinds, chan_modes, methods, dists):
        feats = feat_map[hk]
        per_cut, avg = loo_score(hk, feats, cm, mt, ds)
        rows.append(dict(hi=hk, chan=cm, method=mt, dist=ds,
                         cut40=per_cut[40], cut50=per_cut[50], cut60=per_cut[60],
                         avg=avg))
    res = pd.DataFrame(rows).sort_values("avg", ascending=False).reset_index(drop=True)
    return res


def predict_validation(hi_kind, feats, chan_mode, method, dist,
                       smooth=5, temp=0.5, censor_extrap=False):
    """Predict the 6 validation bearings using ALL 4 training bearings cut@50."""
    lib = []
    for tid in TRAIN_IDS:
        hi = train_hi_cut(tid, 50, hi_kind, feats, chan_mode, smooth)
        lib.append((tid, hi, rul_at(TRAIN_N[tid], 50)))
    preds = {}
    diag = {}
    for tid in TEST_IDS:
        test_hi = test_hi_series(tid, hi_kind, feats, chan_mode, smooth)
        pred, cands = predict_one(test_hi, lib, method, dist, temp)
        if censor_extrap:
            pred = apply_censor(test_hi, lib, pred, 50)
        preds[tid] = pred
        diag[tid] = cands
    return preds, diag


if __name__ == "__main__":
    print("=== RUL@cut reference values ===")
    for tid in TRAIN_IDS:
        print(f"Train{tid}: N={TRAIN_N[tid]} life={life_total(TRAIN_N[tid]):.0f} "
              f"RUL@40={rul_at(TRAIN_N[tid],40):.0f} RUL@50={rul_at(TRAIN_N[tid],50):.0f} "
              f"RUL@60={rul_at(TRAIN_N[tid],60):.0f}")

    print("\n=== GRID SEARCH (LOO) ===")
    res = grid_search()
    res.to_csv(f"{OUT}/loo_grid.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(res.head(15).to_string(index=False))

    best = res.iloc[0]
    print(f"\n=== BEST CONFIG: {dict(best)} ===")
    feat_map = {"zall": FEATS6, "ztransfer": FEATS_TRANSFER, "pca": FEATS6}
    bf = feat_map[best.hi]

    # Validation predictions with best config (no censor and with censor)
    for censor in [False, True]:
        preds, diag = predict_validation(best.hi, bf, best.chan, best.method,
                                         best.dist, censor_extrap=censor)
        anchor = [59351, 51138, 19936, 12168, 1800, 1800]
        print(f"\n--- Validation predictions (censor={censor}) ---")
        for i, tid in enumerate(TEST_IDS):
            print(f"Val{tid}: {preds[tid]:.0f}")
        pv = [preds[t] for t in TEST_IDS]
        print("int:", [int(round(x)) for x in pv])

    # Val2 raw with several top configs
    print("\n=== Val2 raw across top-5 configs ===")
    for _, r in res.head(5).iterrows():
        ff = feat_map[r.hi]
        preds, _ = predict_validation(r.hi, ff, r.chan, r.method, r.dist)
        print(f"{r.hi}/{r.chan}/{r.method}/{r.dist}: "
              f"Val2={preds[2]:.0f}  all={[int(round(preds[t])) for t in TEST_IDS]}")
