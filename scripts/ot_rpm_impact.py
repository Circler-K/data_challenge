"""Quantify how much ESTIMATED RPM (vibration-only, test-faithful) degrades the
A_RUL score versus TRUE RPM (operation.csv) — the #1 optimism in our self-
validation.

For each Train, builds order-tracking features TWICE (same fast extractor, no
VMD): once with operation-CSV RPM, once with rpm_estimator's vibration-only
estimate. Both feature sets go through the identical LOTO + cut + official
A_RUL harness, so the delta isolates the RPM-estimation penalty.

Run:  python scripts/ot_rpm_impact.py
Outputs: outputs/ot_features/{true,est}/Train{i}.csv  + console A_RUL table.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert, savgol_filter
from scipy.stats import kurtosis

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
INHWAN = Path("c:/Users/User/WorkSpace/INHWAN")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(INHWAN))

import rpm_estimator as R  # noqa: E402
from src.scoring import a_rul_score, error_pct  # noqa: E402
from src.operation import load_operation, align_to_vibration, list_vibration_files  # noqa: E402

FS = 25600
SAMPLES_PER_REV = 1024
CHS = (0, 1, 2, 3)
OUT = ROOT / "outputs" / "ot_features"
CUT_FRACS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
KNEE_FRAC = 0.30
ASYM_ALPHA = 1.15
HEALTHY_FRAC = 0.10
ROLL_WINDOWS = (6, 18)


# ---------------- order tracking + features (no VMD) ----------------
def order_track(sig: np.ndarray, rpm: float) -> np.ndarray:
    """Angular resample one channel at a constant file RPM."""
    n = len(sig)
    t = np.arange(n) / FS
    rpm_up = np.full(n, max(rpm, 1.0))
    phase = cumulative_trapezoid(rpm_up / 60.0, t, initial=0.0)
    uphase, uidx = np.unique(phase, return_index=True)
    max_rev = uphase[-1]
    m = int(max_rev * SAMPLES_PER_REV)
    if m < 16:
        return sig.astype(float)
    even = np.linspace(0, max_rev, m)
    f = interp1d(uphase, sig[uidx], kind="linear", bounds_error=False, fill_value=0.0)
    return f(even)


def ot_features(ot: np.ndarray, ch: int) -> dict:
    p = f"Ch{ch}_"
    rms = float(np.sqrt(np.mean(ot ** 2)))
    out = {p + "OT_RMS": rms,
           p + "OT_Kurtosis": float(kurtosis(ot, fisher=True, bias=False)),
           p + "OT_CrestFactor": float(np.max(np.abs(ot)) / (rms + 1e-9))}
    spec = np.abs(np.fft.rfft(ot))
    out[p + "Order_BandEnergy"] = float(np.log10(np.sum(spec[1:50] ** 2) + 1.0))
    psd = spec[1:] ** 2
    pn = psd / (np.sum(psd) + 1e-12)
    out[p + "Spectral_Entropy"] = float(-np.sum(pn * np.log(pn + 1e-12)))
    env = np.abs(hilbert(ot))
    es = np.abs(np.fft.rfft(env - env.mean()))
    out[p + "Env_BandEnergy"] = float(np.log10(np.sum(es[1:50] ** 2) + 1.0))
    return out


# ---------------- extraction (both RPM sources) ----------------
def extract_train(tr: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = list_vibration_files(tr)
    idxs = [int(f.stem) for f in files]

    # true RPM per file (operation.csv)
    op = load_operation(tr)
    agg = align_to_vibration(op, len(files))
    true_rpm = {i: float(agg.loc[agg.file_idx == k, "rpm_mean"].iloc[0])
                for k, i in enumerate(idxs, start=1)}

    # estimated RPM per file (vibration-only): raw harmonic-sum then stepwise refine
    ch0 = []
    for f in files:
        chs = TdmsFile.read(str(f)).groups()[0].channels()
        ch0.append(chs[0][:])
    raw = R.estimate_rpm_series(ch0)
    est = R.refine_stepwise(raw)
    est_rpm = {i: float(est[k]) for k, i in enumerate(idxs)}

    rows_true, rows_est = [], []
    for k, f in enumerate(files):
        i = idxs[k]
        sigs = [c[:] for c in TdmsFile.read(str(f)).groups()[0].channels()][:4]
        base = dict(File_Index=i, train_id=tr)
        rt, re_ = dict(base), dict(base)
        for ch in CHS:
            if ch < len(sigs):
                rt.update(ot_features(order_track(sigs[ch], true_rpm[i]), ch))
                re_.update(ot_features(order_track(sigs[ch], est_rpm[i]), ch))
        rows_true.append(rt)
        rows_est.append(re_)
    return pd.DataFrame(rows_true), pd.DataFrame(rows_est)


# ---------------- harness (mirrors simulate_arul_ot) ----------------
def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["train_id", "File_Index"]).reset_index(drop=True)
    df["t_start_sec"] = (df["File_Index"] - 1) * 600
    parts = []
    for tr, sub in df.groupby("train_id"):
        sub = sub.copy()
        eol = sub["t_start_sec"].max() + 60
        sub["time_to_eol_sec"] = eol - sub["t_start_sec"]
        sub["life_frac"] = sub["t_start_sec"] / max(eol, 1)
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    # derived (causal, vibration-only)
    g = df.groupby("train_id", sort=False)
    base = [f"Ch{c}_OT_RMS" for c in CHS] + [f"Ch{c}_OT_Kurtosis" for c in CHS] + \
           [f"Ch{c}_Env_BandEnergy" for c in CHS]
    for col in base:
        for w in ROLL_WINDOWS:
            df[f"{col}_rm{w}"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
            df[f"{col}_rs{w}"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
    for tr in df.train_id.unique():
        m = (df.train_id == tr).to_numpy()
        n_h = max(2, int(m.sum() * HEALTHY_FRAC))
        cc = [f"Ch{c}_OT_RMS" for c in CHS] + [f"Ch{c}_Env_BandEnergy" for c in CHS]
        sub = df.loc[m, cc]
        cs = (sub - sub.iloc[:n_h].mean()).cumsum()
        for col in cc:
            df.loc[m, f"{col}_cs"] = cs[col].values
    return df


def asym(yt, yp):
    e = yp - yt
    w = np.where(e > 0, ASYM_ALPHA, 1.0)
    return 2 * w * e, 2 * w


def score_set(df: pd.DataFrame, mode: str) -> float:
    df = prep(df)
    rul = df["time_to_eol_sec"].astype(float)
    if mode == "capped":
        total = df.groupby("train_id")["time_to_eol_sec"].transform("max").astype(float)
        rul = rul.clip(upper=total * KNEE_FRAC)
    df["RUL"] = rul
    drop = {"File_Index", "train_id", "t_start_sec", "time_to_eol_sec", "life_frac", "RUL"}
    cols = [c for c in df.columns if c not in drop]
    eol = {tr: float(s["t_start_sec"].max() + 60) for tr, s in df.groupby("train_id")}
    sc = []
    for held in sorted(df.train_id.unique()):
        m = df.train_id != held
        mdl = lgb.LGBMRegressor(objective=asym, learning_rate=0.05, num_leaves=31,
                                max_depth=6, feature_fraction=0.8, random_state=42,
                                verbose=-1, n_estimators=300)
        mdl.fit(df.loc[m, cols], df.loc[m, "RUL"])
        sub = df[df.train_id == held].sort_values("File_Index").reset_index(drop=True)
        n = len(sub)
        pred = mdl.predict(sub[cols])
        for fr in CUT_FRACS:
            cut = min(n - 1, max(1, int(round(fr * n)) - 1))
            act = eol[held] - float(sub.loc[cut, "t_start_sec"])
            seg = pred[: cut + 1].copy()
            w = min(51, len(seg) // 2 * 2 - 1)
            seg = savgol_filter(seg, w, 1) if w >= 3 else seg
            sc.append(float(a_rul_score(act, float(np.minimum.accumulate(seg)[-1]))))
    return float(np.mean(sc))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dft, dfe = [], []
    for tr in (1, 2, 3, 4):
        t0 = time.time()
        a, b = extract_train(tr)
        (OUT / "true").mkdir(exist_ok=True)
        (OUT / "est").mkdir(exist_ok=True)
        a.to_csv(OUT / "true" / f"Train{tr}.csv", index=False)
        b.to_csv(OUT / "est" / f"Train{tr}.csv", index=False)
        dft.append(a)
        dfe.append(b)
        print(f"Train{tr} extracted ({len(a)} files, {time.time()-t0:.1f}s)")
    dft = pd.concat(dft, ignore_index=True)
    dfe = pd.concat(dfe, ignore_index=True)
    print("\n=== A_RUL: TRUE rpm vs ESTIMATED rpm (vibration-only) ===")
    for mode in ("raw", "capped"):
        st = score_set(dft.copy(), mode)
        se = score_set(dfe.copy(), mode)
        print(f"  mode={mode:6s}  true_rpm={st:.4f}   est_rpm={se:.4f}   "
              f"drop={st-se:+.4f}")


if __name__ == "__main__":
    main()
