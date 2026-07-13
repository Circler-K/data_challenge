"""A_RUL harness variant using INHWAN order-tracking features.

Answers empirically: "does an order-tracking feature frontend beat the
kurtogram-envelope features under the REAL A_RUL metric?"

Same LOTO + cut + official scoring as simulate_arul.py, but the feature source
is INHWAN/Train{i}_order_tracking.csv (4 ch x 7 OT features). Runs both raw and
capped RUL targets so the numbers line up directly with the envelope baseline
(raw 0.15 / capped 0.41).

NOTE: these OT csvs were generated with operation-CSV RPM (file-matched). At
real test time RPM must come from rpm_estimator.py, so this is an *upper bound*
on OT-feature quality — but a fair feature-vs-feature comparison.

Run:  python scripts/simulate_arul_ot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
INHWAN = Path("c:/Users/User/WorkSpace/INHWAN")
sys.path.insert(0, str(ROOT))
from src.scoring import a_rul_score, error_pct  # noqa: E402

CHS = ("Ch0", "Ch1", "Ch2", "Ch3")
OT_FEATS = ("OT_RMS", "OT_Kurtosis", "OT_CrestFactor", "Order_BandEnergy",
            "Spectral_Entropy", "Env_BandEnergy", "VMD_IMF1_RMS")
ENERGY_FEATS = ("Order_BandEnergy", "Env_BandEnergy")  # log-transform these
CUT_FRACS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
ROLL_WINDOWS = (6, 18)
HEALTHY_FRAC = 0.10
KNEE_FRAC = 0.30
ASYM_ALPHA = 1.15
SAVGOL_WINDOW, SAVGOL_POLY = 51, 1
FILE_PERIOD = 600  # sec


def load_ot() -> pd.DataFrame:
    parts = []
    for tr in (1, 2, 3, 4):
        df = pd.read_csv(INHWAN / f"Train{tr}_order_tracking.csv")
        df = df.sort_values("File_Index").reset_index(drop=True)
        df["train_id"] = tr
        df["t_start_sec"] = (df["File_Index"] - 1) * FILE_PERIOD
        eol = (df["File_Index"].max() - 1) * FILE_PERIOD + 60
        df["time_to_eol_sec"] = eol - df["t_start_sec"]
        df["life_frac"] = df["t_start_sec"] / max(eol, 1)
        # log-transform blown-up energy features
        for ch in CHS:
            for f in ENERGY_FEATS:
                c = f"{ch}_{f}"
                if c in df.columns:
                    df[c] = np.log10(df[c].clip(lower=0) + 1.0)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def clip_per_train(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for tr in out["train_id"].unique():
        m = (out["train_id"] == tr).to_numpy()
        for ch in CHS:
            kc = f"{ch}_OT_Kurtosis"
            if kc in out.columns:
                out.loc[m, kc] = out.loc[m, kc].clip(upper=10.0)
            for f in ("OT_RMS", "OT_CrestFactor", "Order_BandEnergy",
                      "Env_BandEnergy", "VMD_IMF1_RMS"):
                c = f"{ch}_{f}"
                if c in out.columns:
                    cap = float(np.nanpercentile(out.loc[m, c], 99))
                    out.loc[m, c] = out.loc[m, c].clip(upper=cap)
    return out


def _ols_slope(arr):
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    xm = x.mean()
    d = ((x - xm) ** 2).sum()
    return 0.0 if d == 0 else float(((x - xm) * (arr - arr.mean())).sum() / d)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("train_id", sort=False)
    roll_cols = [f"{ch}_{f}" for ch in CHS
                 for f in ("OT_RMS", "OT_Kurtosis", "Env_BandEnergy")]
    for col in roll_cols:
        for w in ROLL_WINDOWS:
            mean = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
            std = g[col].transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
            out[f"{col}_roll_mean_{w}"] = mean
            out[f"{col}_roll_std_{w}"] = std
            out[f"{col}_trend_{w}"] = out[col] - mean
    for ch in CHS:
        col = f"{ch}_OT_RMS"
        out[f"{col}_slope_1h"] = g[col].transform(
            lambda s: s.rolling(6, min_periods=2).apply(_ols_slope, raw=True).fillna(0.0))
    cusum_cols = [f"{ch}_OT_RMS" for ch in CHS] + [f"{ch}_Env_BandEnergy" for ch in CHS]
    for tr in out["train_id"].unique():
        m = (out["train_id"] == tr).to_numpy()
        n_h = max(2, int(m.sum() * HEALTHY_FRAC))
        sub = out.loc[m, cusum_cols]
        mu = sub.iloc[:n_h].mean()
        cs = (sub - mu).cumsum()
        for col in cusum_cols:
            out.loc[m, f"{col}_cusum"] = cs[col].values
    return out


def feature_columns(df):
    drop = {"File_Index", "Target_Set", "train_id", "t_start_sec",
            "time_to_eol_sec", "life_frac", "RUL_sec"}
    return [c for c in df.columns if c not in drop]


def build_rul(df, mode):
    rul = df["time_to_eol_sec"].astype(np.float64)
    if mode == "raw":
        return rul
    total = df.groupby("train_id")["time_to_eol_sec"].transform("max").astype(np.float64)
    return rul.clip(upper=total * KNEE_FRAC)


def asym(y_true, y_pred):
    err = y_pred - y_true
    w = np.where(err > 0, ASYM_ALPHA, 1.0)
    return 2.0 * w * err, 2.0 * w


def fit(X, y):
    m = lgb.LGBMRegressor(objective=asym, metric="rmse", learning_rate=0.05,
                          num_leaves=31, max_depth=6, feature_fraction=0.8,
                          random_state=42, verbose=-1, n_estimators=300)
    m.fit(X, y)
    return m


def run(mode):
    df = clip_per_train(load_ot())
    df["RUL_sec"] = build_rul(df, mode)
    df = add_derived(df)
    cols = feature_columns(df)
    eol = {tr: float(sub["t_start_sec"].max() + 60)
           for tr, sub in df.groupby("train_id")}
    rows = []
    for held in (1, 2, 3, 4):
        tr_mask = df.train_id != held
        model = fit(df.loc[tr_mask, cols], df.loc[tr_mask, "RUL_sec"])
        sub = df[df.train_id == held].sort_values("File_Index").reset_index(drop=True)
        n = len(sub)
        pred_full = model.predict(sub[cols])
        for f in CUT_FRACS:
            cut = min(n - 1, max(1, int(round(f * n)) - 1))
            act = eol[held] - float(sub.loc[cut, "t_start_sec"])
            pred_raw = float(pred_full[cut])
            seg = pred_full[: cut + 1].copy()
            w = min(SAVGOL_WINDOW, len(seg) // 2 * 2 - 1)
            seg_s = savgol_filter(seg, w, SAVGOL_POLY) if w >= SAVGOL_POLY + 2 else seg
            pred_mono = float(np.minimum.accumulate(seg_s)[-1])
            rows.append(dict(held=held, cut_frac=f, act_h=act / 3600.0,
                             pred_mono_h=pred_mono / 3600.0,
                             er_mono=float(error_pct(act, pred_mono)),
                             score_raw=float(a_rul_score(act, pred_raw)),
                             score_mono=float(a_rul_score(act, pred_mono))))
    return pd.DataFrame(rows)


def main():
    for mode in ("raw", "capped"):
        res = run(mode)
        pd.set_option("display.width", 150)
        pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
        print("\n" + "=" * 80)
        print(f" ORDER-TRACKING features - A_RUL (mode={mode})")
        print("=" * 80)
        print(res.groupby("cut_frac")[["score_raw", "score_mono"]].mean().to_string())
        print(res.groupby("held")[["score_raw", "score_mono"]].mean().to_string())
        print(f">>> OVERALL  raw={res.score_raw.mean():.4f}  "
              f"monotone={res.score_mono.mean():.4f}")
        res.to_csv(ROOT / "outputs" / f"arul_sim_ot_{mode}.csv", index=False)


if __name__ == "__main__":
    main()
