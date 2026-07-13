"""A_RUL simulation harness — the ONLY metric that mirrors the real challenge.

Why this exists
---------------
The challenge scores a single RUL prediction per bearing, taken at the LAST
timestamp of a validation segment that is cut *before* failure, using the
official A_RUL metric (see src/scoring.py). MAE/RMSE on a holdout do NOT
reflect this. This harness reproduces the real test condition:

  1. Leave-One-Train-Out: train on 3 Trains, hold out the 4th.
  2. "Cut" the held-out Train at several fractions of its life (it ran to
     failure, but we pretend we only saw up to the cut — exactly like the
     validation segments).
  3. Predict RUL at the cut file using a VIBRATION-ONLY model (no RPM / torque /
     temperature — those do not exist at test time).
  4. Score with the official A_RUL against the true seconds-to-failure.

Output: per-(held Train, cut fraction) table + mean final score, a CSV, and a
diagnostic plot of A_RUL vs cut fraction.

Run:
    python scripts/simulate_arul.py
    python scripts/simulate_arul.py --rul-mode capped   # piecewise-knee target
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_features import CHANNELS, ENV_BANDS, clip_features  # noqa: E402
from src.scoring import a_rul_score, error_pct  # noqa: E402

FEAT_DIR = ROOT / "outputs" / "features_utils"
FIG_DIR = ROOT / "outputs" / "figures"

# Cut points: fraction of the held-out Train's life that we "see" before the
# segment ends. The validation segments are cut while degradation is in
# progress, so we probe several plausible cut depths.
CUT_FRACS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

ROLL_WINDOWS = (6, 18)     # files; 1 file = 10 min => 1h, 3h
SLOPE_WINDOW = 6           # 1h
HEALTHY_FRAC = 0.10        # first 10% of a segment = healthy reference
KNEE_FRAC = 0.30           # for --rul-mode capped: plateau = total_life * 0.30
ASYM_ALPHA = 1.15          # over-prediction penalty in the custom loss
SAVGOL_WINDOW = 51
SAVGOL_POLY = 1


# ----------------------------------------------------------------------------
# Vibration-only feature set
# ----------------------------------------------------------------------------
def vib_base_columns(df: pd.DataFrame) -> list[str]:
    """All raw vibration features (everything that starts with CH)."""
    return [c for c in df.columns if c.startswith("CH")]


def clip_per_train(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clip_features() within each Train (mirrors clipping a test unit on
    its own statistics — no cross-unit leakage)."""
    parts = []
    for tr, sub in df.groupby("train_id", sort=False):
        parts.append(clip_features(sub.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


def add_vib_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Causal, vibration-only derived features (computed per Train so windows
    never bleed across bearings). All look-back only => safe to compute on the
    full trajectory and slice at the cut."""
    out = df.copy()
    g = out.groupby("train_id", sort=False)

    roll_cols = [f"{ch}_rms" for ch in CHANNELS]
    roll_cols += [f"{ch}_kurt" for ch in CHANNELS]
    roll_cols += [f"{ch}_env_BSF_1x" for ch in CHANNELS]
    for col in roll_cols:
        for w in ROLL_WINDOWS:
            mean = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
            std = g[col].transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
            out[f"{col}_roll_mean_{w}"] = mean
            out[f"{col}_roll_std_{w}"] = std
            out[f"{col}_trend_{w}"] = out[col] - mean

    # slope over last hour on each channel RMS
    for ch in CHANNELS:
        col = f"{ch}_rms"
        out[f"{col}_slope_1h"] = g[col].transform(
            lambda s: s.rolling(SLOPE_WINDOW, min_periods=2)
                       .apply(_ols_slope, raw=True).fillna(0.0))

    # cumulative departure (CUSUM) from per-Train healthy mean
    cusum_cols = [f"{ch}_rms" for ch in CHANNELS] + [f"{ch}_env_BSF_1x" for ch in CHANNELS]
    for tr in out["train_id"].unique():
        mask = (out["train_id"] == tr).to_numpy()
        n_h = max(2, int(mask.sum() * HEALTHY_FRAC))
        sub = out.loc[mask, cusum_cols]
        mu = sub.iloc[:n_h].mean()
        cusum = (sub - mu).cumsum()
        for col in cusum_cols:
            out.loc[mask, f"{col}_cusum"] = cusum[col].values

    # Mahalanobis-style anomaly on the 36 envelope bands vs healthy baseline
    env_cols = [f"{ch}_env_{b}" for ch in CHANNELS for b in ENV_BANDS]
    scores = np.zeros(len(out))
    for tr in out["train_id"].unique():
        mask = (out["train_id"] == tr).to_numpy()
        arr = out.loc[mask, env_cols].to_numpy()
        n_h = max(2, int(len(arr) * HEALTHY_FRAC))
        mu = arr[:n_h].mean(axis=0)
        sigma = arr[:n_h].std(axis=0) + 1e-8
        diff = (arr - mu) / sigma
        scores[mask] = np.sqrt((diff ** 2).sum(axis=1))
    out["anomaly_score"] = scores
    return out


def _ols_slope(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (arr - arr.mean())).sum() / denom)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Final model feature list: vibration base + vibration-derived, never any
    operation column or time/index column."""
    drop = {"train_id", "file_idx", "file_name", "t_start_sec",
            "time_to_eol_sec", "life_frac", "RUL_sec"}
    op_prefixes = ("rpm_", "torque_", "tcf_", "tcr_")
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if any(c.startswith(p) for p in op_prefixes):
            continue
        cols.append(c)
    return cols


# ----------------------------------------------------------------------------
# RUL target
# ----------------------------------------------------------------------------
def build_rul_sec(df: pd.DataFrame, mode: str) -> pd.Series:
    """True seconds-to-failure, optionally capped at a per-Train knee."""
    rul = df["time_to_eol_sec"].astype(np.float64)
    if mode == "raw":
        return rul
    total = df.groupby("train_id")["time_to_eol_sec"].transform("max").astype(np.float64)
    return rul.clip(upper=total * KNEE_FRAC)


def asymmetric_mse(y_true, y_pred):
    """Custom MSE penalizing over-prediction (pred > true) by ASYM_ALPHA."""
    err = y_pred - y_true
    w = np.where(err > 0, ASYM_ALPHA, 1.0)
    return 2.0 * w * err, 2.0 * w


# ----------------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------------
def load_all() -> pd.DataFrame:
    parts = []
    for tr in (1, 2, 3, 4):
        df = pd.read_parquet(FEAT_DIR / f"train{tr}.parquet")
        df = df.sort_values("file_idx").reset_index(drop=True)
        if "train_id" not in df.columns:
            df["train_id"] = tr
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def fit_model(X, y):
    params = dict(
        objective=asymmetric_mse, metric="rmse", boosting_type="gbdt",
        learning_rate=0.05, num_leaves=31, max_depth=6,
        feature_fraction=0.8, random_state=42, verbose=-1,
    )
    model = lgb.LGBMRegressor(**params, n_estimators=300)
    model.fit(X, y)
    return model


def run(rul_mode: str) -> pd.DataFrame:
    df = load_all()
    df = clip_per_train(df)
    df["RUL_sec"] = build_rul_sec(df, rul_mode)
    df = add_vib_derived(df)
    feat_cols = feature_columns(df)
    print(f"vibration-only feature count: {len(feat_cols)}")

    # per-Train EOL (seconds) and life length
    eol = {}
    for tr, sub in df.groupby("train_id"):
        last_start = sub["t_start_sec"].max()
        eol[tr] = float(last_start + 60)  # last file start + 60s capture

    rows = []
    for held in (1, 2, 3, 4):
        train_mask = df.train_id != held
        Xtr, ytr = df.loc[train_mask, feat_cols], df.loc[train_mask, "RUL_sec"]
        model = fit_model(Xtr, ytr)

        sub = df[df.train_id == held].sort_values("file_idx").reset_index(drop=True)
        n = len(sub)
        # full-trajectory predictions (for the smoothed/monotone variant)
        pred_full = model.predict(sub[feat_cols])

        for f in CUT_FRACS:
            cut = min(n - 1, max(1, int(round(f * n)) - 1))
            t_cut = float(sub.loc[cut, "t_start_sec"])
            act = eol[held] - t_cut                       # true seconds-to-failure
            pred_raw = float(pred_full[cut])

            # smoothed + monotone variant, evaluated up to the cut only
            seg = pred_full[: cut + 1].copy()
            w = min(SAVGOL_WINDOW, len(seg) // 2 * 2 - 1)
            seg_s = (savgol_filter(seg, w, SAVGOL_POLY)
                     if w >= SAVGOL_POLY + 2 else seg)
            pred_mono = float(np.minimum.accumulate(seg_s)[-1])

            rows.append(dict(
                held=held, cut_frac=f, cut_idx=cut + 1, n_files=n,
                act_h=act / 3600.0,
                pred_raw_h=pred_raw / 3600.0,
                pred_mono_h=pred_mono / 3600.0,
                er_raw=float(error_pct(act, pred_raw)),
                er_mono=float(error_pct(act, pred_mono)),
                score_raw=float(a_rul_score(act, pred_raw)),
                score_mono=float(a_rul_score(act, pred_mono)),
            ))
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, rul_mode: str) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    print("\n" + "=" * 96)
    print(f" A_RUL SIMULATION  (vibration-only model, rul_mode={rul_mode})")
    print("=" * 96)
    show = res[["held", "cut_frac", "cut_idx", "n_files",
                "act_h", "pred_raw_h", "er_raw", "score_raw",
                "pred_mono_h", "er_mono", "score_mono"]]
    print(show.to_string(index=False))

    print("\n--- mean A_RUL by cut fraction (this is what the challenge averages) ---")
    by_cut = res.groupby("cut_frac")[["score_raw", "score_mono"]].mean()
    print(by_cut.to_string())

    print("\n--- mean A_RUL by held-out Train ---")
    by_held = res.groupby("held")[["score_raw", "score_mono"]].mean()
    print(by_held.to_string())

    print(f"\n>>> OVERALL mean A_RUL  raw={res.score_raw.mean():.4f}  "
          f"monotone={res.score_mono.mean():.4f}")
    print("    (1.0 = perfect; 0.5 = 20% over- or 50% under-prediction)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ROOT / "outputs" / f"arul_sim_{rul_mode}.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for held, sub in res.groupby("held"):
        ax.plot(sub.cut_frac, sub.score_mono, marker="o",
                label=f"Train{held} (monotone)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("cut fraction of life (validation segment depth)")
    ax.set_ylabel("A_RUL score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"A_RUL vs cut depth — vibration-only LOTO ({rul_mode} RUL)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out_fig = FIG_DIR / f"arul_sim_{rul_mode}.png"
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rul-mode", choices=["raw", "capped"], default="raw",
                   help="raw = true seconds-to-failure; capped = piecewise knee")
    args = p.parse_args()
    res = run(args.rul_mode)
    report(res, args.rul_mode)


if __name__ == "__main__":
    main()
