"""RUL prediction (LightGBM) on outputs/features_utils/train{1..4}.parquet.

Adapts the user's piecewise-linear RUL template to our actual schema.

Schema mapping (template → ours):
  CH1_RMS         → CH{1..4}_rms
  CH1_Kurtosis    → CH{1..4}_kurt
  Front_Temp      → tcf_max  (+ tcr_max for symmetry)
  BSF_1x_amp      → CH{1..4}_env_BSF_1x

Pipeline:
  1. Load 4 parquets, concat (train_id preserved)
  2. clip_features(): Kurt cap=10, RMS/env at p99
  3. RUL_h = time_to_eol_sec / 3600;
     piecewise: RUL = min(RUL_h, total_h * 0.7) per Train
  4. Rolling features (per Train, windows 3 & 10) on key columns
  5. Train [1,2,3] → Validation [4]
  6. LightGBM regression, early-stopping on val RMSE
  7. Outputs: outputs/figures/rul_pred_train4.png + rul_importance.png
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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT / "scripts"))
from plot_features import CHANNELS, clip_features  # noqa: E402

FEAT_DIR = ROOT / "outputs" / "features_utils"
FIG_DIR = ROOT / "outputs" / "figures"

ROLL_WINDOWS = (6, 18)  # files = 60 min, 180 min (1 file = 10 min capture)
SLOPE_WINDOW = 6  # files = 1 hour (10-min capture period)
KNEE_FRAC = 0.30  # plateau value = total_h * 0.30 → FPT at 70 % of life elapsed
TRAIN_IDS = (1, 2, 4)  # Leave-One-Train-Out: Train3 held out
VALID_IDS = (3,)
ASYM_ALPHA = 1.15    # mild conservative nudge (was 3.0 — too aggressive)
DECLINE_WEIGHT = 3.0  # sample-weight for rows in decline phase (RUL < plateau)
TRANSITION_LIFE_FRAC = (0.65, 0.95)   # life_frac window covering FPT + early decline
TRANSITION_WEIGHT = 5.0               # extra weight on the transition zone
SAVGOL_WINDOW = 51   # odd; ~510 min — flattens the 2.5h V-trap (was 11)
SAVGOL_POLY = 1      # linear fit — bluntest shape (was 2)
DEGRADE_HEALTHY_FRAC = 0.30  # baseline window for is_degrading threshold
DEGRADE_MULTIPLIER = 1.5     # 1.5× early-life max — was 2.0, too strict
PCA_HEALTHY_FRAC = 0.30  # use first 30 % of life ("0~9h" for Train3) as healthy ref
PCA_N_COMPONENTS = 5
ENV_BAND_NAMES = (
    "BPFI_1x", "BPFI_2x", "BPFI_3x",
    "BPFO_1x", "BPFO_2x", "BPFO_3x",
    "BSF_1x",  "BSF_2x",  "BSF_3x",
)
HEALTHY_FRAC = 0.10  # first 10 % of each Train = healthy reference


def base_roll_columns() -> list[str]:
    cols = []
    for ch in CHANNELS:
        cols += [f"{ch}_rms", f"{ch}_kurt", f"{ch}_env_BSF_1x"]
    cols += ["tcf_max", "tcr_max"]
    return cols


def add_rolling_features(df: pd.DataFrame,
                         windows: tuple[int, ...] = ROLL_WINDOWS) -> pd.DataFrame:
    """Compute rolling mean/std/trend per Train (groupby) so windows don't
    bleed across train boundaries."""
    df_out = df.copy()
    cols = [c for c in base_roll_columns() if c in df.columns]
    g = df_out.groupby("train_id", sort=False)
    for col in cols:
        for w in windows:
            mean = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
            std = g[col].transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
            df_out[f"{col}_roll_mean_{w}"] = mean
            df_out[f"{col}_roll_std_{w}"] = std
            df_out[f"{col}_trend_{w}"] = df_out[col] - mean
    return df_out


def _ols_slope(arr: np.ndarray) -> float:
    """Slope of best-fit line through `arr` (units = arr per file index)."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (arr - arr.mean())).sum() / denom)


def add_slope_features(df: pd.DataFrame,
                       window: int = SLOPE_WINDOW) -> pd.DataFrame:
    """Linear-regression slope over the last `window` files for RMS *and*
    Front/Rear temperatures.

    Window=6 covers the most recent 1 hour (10-min capture period). Tree models
    cannot extrapolate beyond seen RUL labels — exposing the *rate* of RMS or
    temp growth lets the model translate "steeper slope → shorter life
    remaining". Temperature derivatives (tcf/tcr) capture the bearing-friction
    rise that often precedes the vibration spike.
    """
    df_out = df.copy()
    g = df_out.groupby("train_id", sort=False)
    cols = [f"{ch}_rms" for ch in CHANNELS] + ["tcf_max", "tcr_max"]
    for col in cols:
        df_out[f"{col}_slope_1h"] = g[col].transform(
            lambda s: s.rolling(window, min_periods=2)
                       .apply(_ols_slope, raw=True).fillna(0.0)
        )
    return df_out


def asymmetric_mse_objective(y_true: np.ndarray, y_pred: np.ndarray):
    """Custom MSE that penalizes overestimating RUL by ASYM_ALPHA×.

    Over-prediction (pred > actual) means we expect more life than reality —
    machine fails before maintenance, dangerous. Under-prediction is just
    early/conservative maintenance. Asymmetric penalty pushes the model toward
    the safe (under-prediction) side.
    """
    error = y_pred - y_true  # positive = overestimate
    weight = np.where(error > 0, ASYM_ALPHA, 1.0)
    grad = 2.0 * weight * error
    hess = 2.0 * weight
    return grad, hess


def compute_sample_weights(rul: np.ndarray, train_id: np.ndarray,
                           life_frac: np.ndarray,
                           decline_weight: float = DECLINE_WEIGHT,
                           transition_weight: float = TRANSITION_WEIGHT) -> np.ndarray:
    """Higher weight on decline-phase rows AND extra on the FPT/early-decline
    transition zone (life_frac ∈ TRANSITION_LIFE_FRAC).

    Rationale: plateau (60-70 % of rows) and full-failure (~last few rows) are
    both "easy". The transition (10-13h elapsed for Train3) carries the most
    diagnostic info about *when* RUL starts dropping — up-weight it so the
    model fits the FPT slope precisely.
    """
    weights = np.ones(len(rul), dtype=np.float64)
    plateau = pd.Series(rul).groupby(pd.Series(train_id)).transform("max").to_numpy()
    weights[rul < plateau] = decline_weight
    lo, hi = TRANSITION_LIFE_FRAC
    in_transition = (life_frac >= lo) & (life_frac <= hi)
    weights[in_transition] = np.maximum(weights[in_transition], transition_weight)
    return weights


def add_cumulative_load_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative mechanical load proxy = Σ |rpm × torque|. Models bearing
    fatigue accumulation — the physical basis for "older = closer to failure".
    """
    df_out = df.copy()
    instant = df_out["rpm_mean"].abs() * df_out["torque_mean"].abs()
    df_out["cumulative_load"] = (
        instant.groupby(df_out["train_id"]).cumsum().to_numpy()
    )
    return df_out


def add_pca_recon_feature(df: pd.DataFrame,
                          healthy_frac: float = PCA_HEALTHY_FRAC,
                          n_components: int = PCA_N_COMPONENTS) -> pd.DataFrame:
    """PCA reconstruction error using a broad sensor feature set; PCA fit
    only on the per-Train healthy window (first `healthy_frac` of life).

    Captures *correlation-pattern* deviation from the healthy state — fires
    even when individual sensor magnitudes are still near their normal levels.
    Complementary to ``anomaly_score`` (Mahalanobis), which captures magnitude.
    """
    df_out = df.copy()
    feat_cols = (
        [f"{ch}_rms" for ch in CHANNELS] +
        [f"{ch}_kurt" for ch in CHANNELS] +
        [f"{ch}_env_BSF_1x" for ch in CHANNELS] +
        [f"{ch}_env_BPFI_1x" for ch in CHANNELS] +
        [f"{ch}_env_BPFO_1x" for ch in CHANNELS] +
        ["tcf_max", "tcr_max"]
    )
    recon = np.zeros(len(df_out))
    for tr in df_out["train_id"].unique():
        mask = (df_out["train_id"] == tr).to_numpy()
        sub = df_out.loc[mask, feat_cols].to_numpy()
        n_healthy = max(n_components + 2, int(mask.sum() * healthy_frac))
        healthy = sub[:n_healthy]
        mu = healthy.mean(axis=0)
        sigma = healthy.std(axis=0) + 1e-8
        full_z = (sub - mu) / sigma
        pca = PCA(n_components=n_components)
        pca.fit((healthy - mu) / sigma)
        rec = pca.inverse_transform(pca.transform(full_z))
        recon[mask] = np.sqrt(((full_z - rec) ** 2).sum(axis=1))
    df_out["pca_recon_error"] = recon
    return df_out


def add_is_degrading_feature(df: pd.DataFrame,
                             healthy_frac: float = DEGRADE_HEALTHY_FRAC,
                             multiplier: float = DEGRADE_MULTIPLIER) -> pd.DataFrame:
    """Latched binary degradation flag — trips when ANY of 6 variability
    signals exceeds its per-Train early-life max × `multiplier`:

      - tcf_max_roll_std_18  (Front temp jitter)
      - tcr_max_roll_std_18  (Rear temp jitter)
      - CH{1..4}_rms_roll_std_18  (per-channel RMS jitter)

    OR logic = sensitive to whichever sensor degrades first. Latched (stays 1
    once tripped) since degradation is irreversible.

    Domain rationale: bearing failure precursor manifests as *jitter increase*
    in temp/vibration before absolute values spike. A binary "healthy vs sick"
    cue lets the tree model anchor RUL drop on a single discrete decision
    rather than rebuilding the threshold from continuous features each split.

    Requires roll_std_18 features — must run after add_rolling_features().
    """
    df_out = df.copy()
    flag = np.zeros(len(df_out), dtype=np.int8)
    sources = ["tcf_max_roll_std_18", "tcr_max_roll_std_18"] + [
        f"{ch}_rms_roll_std_18" for ch in CHANNELS
    ]
    for tr in df_out["train_id"].unique():
        mask = (df_out["train_id"] == tr).to_numpy()
        n_healthy = max(2, int(mask.sum() * healthy_frac))
        any_trip = np.zeros(int(mask.sum()), dtype=bool)
        for src in sources:
            sub = df_out.loc[mask, src].to_numpy()
            threshold = float(sub[:n_healthy].max()) * multiplier
            any_trip |= sub > threshold
        # Latch — once tripped, stays tripped
        for i in range(1, len(any_trip)):
            if any_trip[i - 1]:
                any_trip[i] = True
        flag[mask] = any_trip.astype(np.int8)
    df_out["is_degrading"] = flag
    return df_out


def add_cusum_features(df: pd.DataFrame,
                       healthy_frac: float = HEALTHY_FRAC) -> pd.DataFrame:
    """Cumulative-sum departure from per-Train healthy reference.

    For each Train, take the mean of the first 10 % of files as the healthy
    reference μ. Then CUSUM_t = Σ_{i≤t} (x_i - μ). Tiny but persistent shifts
    accumulate into a clear upward trend long before the value itself spikes,
    making early-stage degradation visible to a tree model that otherwise only
    fires on threshold crossings.
    """
    df_out = df.copy()
    cols = [f"{ch}_rms" for ch in CHANNELS] + [f"{ch}_env_BSF_1x" for ch in CHANNELS]
    for tr in df_out["train_id"].unique():
        mask = (df_out["train_id"] == tr).to_numpy()
        sub = df_out.loc[mask, cols]
        n_healthy = max(2, int(mask.sum() * healthy_frac))
        mu = sub.iloc[:n_healthy].mean()
        cusum = (sub - mu).cumsum()
        for col in cols:
            df_out.loc[mask, f"{col}_cusum"] = cusum[col].values
    return df_out


def add_anomaly_features(df: pd.DataFrame,
                         healthy_frac: float = HEALTHY_FRAC) -> pd.DataFrame:
    """Mahalanobis distance from per-Train healthy baseline using the 36
    envelope-band features (4 channels × 9 bands).

    Diagonal covariance (independent bands assumption) for numerical stability
    — full covariance is singular when |healthy| < 36. Single scalar per file:
    ``anomaly_score`` ≈ "how far from this Train's early-life envelope state".
    """
    df_out = df.copy()
    env_cols = [f"{ch}_env_{b}" for ch in CHANNELS for b in ENV_BAND_NAMES]
    scores = np.zeros(len(df_out))
    for tr in df_out["train_id"].unique():
        mask = (df_out["train_id"] == tr).to_numpy()
        sub = df_out.loc[mask, env_cols].to_numpy()
        n_healthy = max(2, int(len(sub) * healthy_frac))
        mu = sub[:n_healthy].mean(axis=0)
        sigma = sub[:n_healthy].std(axis=0) + 1e-8
        diff = (sub - mu) / sigma
        scores[mask] = np.sqrt((diff ** 2).sum(axis=1))
    df_out["anomaly_score"] = scores
    return df_out


def build_rul(df: pd.DataFrame, knee_frac: float = KNEE_FRAC) -> pd.Series:
    """Piecewise-linear RUL [hours] per Train.
        rul_h = time_to_eol_sec / 3600
        cap   = total_lifetime_h * knee_frac
        RUL   = min(rul_h, cap)
    """
    rul_h = df["time_to_eol_sec"] / 3600.0
    total_h = df.groupby("train_id")["time_to_eol_sec"].transform("max") / 3600.0
    return rul_h.clip(upper=total_h * knee_frac)


def load_features() -> pd.DataFrame:
    parts = []
    for tr in (1, 2, 3, 4):
        df = pd.read_parquet(FEAT_DIR / f"train{tr}.parquet")
        df = df.sort_values("file_idx").reset_index(drop=True)
        if "train_id" not in df.columns:
            df["train_id"] = tr
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def predict_test():
    """Train on all of Train1-4, predict on Test1-6 (challenge submission).

    Test features were extracted with Train1-4 grand-mean operation values
    substituted (Test data has no operation CSV). Predictions are saved as
    ``outputs/test_rul_predictions.csv`` and visualised as a 6-panel figure.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading 4 train parquets + test_all.parquet ...")
    parts = [load_features()]
    parts.append(pd.read_parquet(FEAT_DIR / "test_all.parquet"))
    df_all = pd.concat(parts, ignore_index=True)
    df_all = df_all.sort_values(["train_id", "file_idx"]).reset_index(drop=True)
    train_mask = df_all.train_id.isin((1, 2, 3, 4)).to_numpy()
    test_mask = df_all.train_id.between(101, 106).to_numpy()
    print(f"  train rows: {train_mask.sum()}  test rows: {test_mask.sum()}")

    print("[2/5] clipping + RUL (train only) + feature engineering ...")
    df_all = clip_features(df_all)
    df_all["RUL"] = np.nan
    df_all.loc[train_mask, "RUL"] = build_rul(df_all[train_mask]).values
    df_all = add_rolling_features(df_all)
    df_all = add_slope_features(df_all)
    df_all = add_cusum_features(df_all)
    df_all = add_cumulative_load_feature(df_all)
    df_all = add_anomaly_features(df_all)
    df_all = add_pca_recon_feature(df_all)
    df_all = add_is_degrading_feature(df_all)

    drop_cols = ["train_id", "file_idx", "file_name",
                 "t_start_sec", "time_to_eol_sec", "life_frac", "RUL"]
    feature_cols = [c for c in df_all.columns if c not in drop_cols]
    print(f"  feature count: {len(feature_cols)}")

    X_train = df_all.loc[train_mask, feature_cols]
    y_train = df_all.loc[train_mask, "RUL"]
    X_test = df_all.loc[test_mask, feature_cols]
    sample_weight = compute_sample_weights(
        y_train.to_numpy(),
        df_all.loc[train_mask, "train_id"].to_numpy(),
        df_all.loc[train_mask, "life_frac"].to_numpy(),
    )

    print("[3/5] training LightGBM on Train1+2+3+4 (no holdout, fixed iters) ...")
    lgb_params = dict(
        objective=asymmetric_mse_objective,
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        feature_fraction=0.8,
        random_state=42,
        verbose=-1,
    )
    # Use ~best_iter from LOOCV runs (≈ 34) as fixed budget — no valid set
    model = lgb.LGBMRegressor(**lgb_params, n_estimators=50)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    print("[4/5] predicting on Test1-6 + applying savgol + monotone ...")
    test_train_ids = df_all.loc[test_mask, "train_id"].to_numpy()
    y_pred_raw = model.predict(X_test)
    y_pred_smooth = y_pred_raw.copy()
    y_pred_mono = y_pred_raw.copy()
    for tr in np.unique(test_train_ids):
        idx = np.where(test_train_ids == tr)[0]
        seg = y_pred_raw[idx]
        w = min(SAVGOL_WINDOW, len(seg) // 2 * 2 - 1)
        seg_s = (savgol_filter(seg, window_length=w, polyorder=SAVGOL_POLY)
                 if w >= SAVGOL_POLY + 2 else seg)
        y_pred_smooth[idx] = seg_s
        y_pred_mono[idx] = np.minimum.accumulate(seg_s)

    out_df = df_all.loc[test_mask, ["train_id", "file_idx", "file_name",
                                    "t_start_sec"]].copy()
    out_df["test_unit"] = out_df["train_id"] - 100
    out_df["rul_raw"] = y_pred_raw
    out_df["rul_savgol"] = y_pred_smooth
    out_df["rul_monotone"] = y_pred_mono
    out_csv = ROOT / "outputs" / "test_rul_predictions.csv"
    out_df[["test_unit", "file_idx", "file_name",
            "rul_raw", "rul_savgol", "rul_monotone"]].to_csv(out_csv, index=False)
    print(f"  saved {out_csv}")
    for tr in sorted(np.unique(test_train_ids)):
        m = out_df.train_id == tr
        unit = tr - 100
        print(f"    Test{unit}: rul start={out_df.loc[m,'rul_monotone'].iloc[0]:.2f}h"
              f"  end={out_df.loc[m,'rul_monotone'].iloc[-1]:.2f}h"
              f"  Δ={out_df.loc[m,'rul_monotone'].iloc[0]-out_df.loc[m,'rul_monotone'].iloc[-1]:.2f}h")

    print("[5/5] plotting 6-panel figure ...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, tr in zip(axes.flat, sorted(np.unique(test_train_ids))):
        m = out_df.train_id == tr
        unit = tr - 100
        t_h = out_df.loc[m, "t_start_sec"].to_numpy() / 3600.0
        ax.plot(t_h, out_df.loc[m, "rul_raw"], color="lightgray", lw=0.9,
                alpha=0.85, label="raw")
        ax.plot(t_h, out_df.loc[m, "rul_savgol"], color="tab:cyan", lw=1.0,
                alpha=0.85, label="savgol")
        ax.plot(t_h, out_df.loc[m, "rul_monotone"], color="tab:red", lw=1.4,
                label="monotone")
        ax.set_title(f"Test{unit}  ({m.sum()} files)", fontsize=11)
        ax.set_xlabel("Time [h]")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Predicted RUL [hours]")
    axes[1, 0].set_ylabel("Predicted RUL [hours]")
    axes[0, 0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Test1-6 RUL Predictions (trained on Train1+2+3+4, "
                 "Test operation = Train mean substitute)", fontsize=12)
    out_fig = FIG_DIR / "rul_pred_test_all.png"
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-train", type=int, choices=[1, 2, 3, 4],
                        help="Use only one Train, temporal 70/30 split")
    parser.add_argument("--predict-test", action="store_true",
                        help="Train on all Train1-4, predict on Test1-6")
    args = parser.parse_args()

    if args.predict_test:
        predict_test()
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading parquets...")
    df_all = load_features()
    if args.single_train:
        df_all = df_all[df_all.train_id == args.single_train].reset_index(drop=True)
    print(f"  total rows: {len(df_all)}  trains: {sorted(df_all.train_id.unique())}")

    print("[2/6] clipping outliers (Kurt cap=10, RMS/env p99)...")
    df_all = clip_features(df_all)

    print("[3/6] building piecewise RUL target (knee at 70% of life)...")
    df_all["RUL"] = build_rul(df_all)
    for tr in sorted(df_all.train_id.unique()):
        m = df_all.train_id == tr
        print(f"  Train{tr}: RUL min={df_all.loc[m,'RUL'].min():.2f}h  "
              f"max={df_all.loc[m,'RUL'].max():.2f}h  "
              f"plateau_rows={(df_all.loc[m,'RUL']==df_all.loc[m,'RUL'].max()).sum()}/{m.sum()}")

    print("[4/6] adding rolling + slope + cusum + load + anomaly + PCA + is_degrading...")
    df_all = add_rolling_features(df_all)
    df_all = add_slope_features(df_all)
    df_all = add_cusum_features(df_all)
    df_all = add_cumulative_load_feature(df_all)
    df_all = add_anomaly_features(df_all)
    df_all = add_pca_recon_feature(df_all)
    df_all = add_is_degrading_feature(df_all)  # depends on tcf_max_roll_std_18
    new_feats = [f"{c}_slope_1h" for c in
                 [f"{ch}_rms" for ch in CHANNELS] + ["tcf_max", "tcr_max"]]
    new_feats += [f"{ch}_rms_cusum" for ch in CHANNELS]
    new_feats += [f"{ch}_env_BSF_1x_cusum" for ch in CHANNELS]
    new_feats += ["cumulative_load", "anomaly_score", "pca_recon_error", "is_degrading"]
    print(f"  added: {new_feats}")
    for tr in sorted(df_all.train_id.unique()):
        m = df_all.train_id == tr
        sub = df_all.loc[m].reset_index(drop=False)
        flags = sub["is_degrading"].to_numpy()
        n_deg = int(flags.sum())
        if n_deg > 0:
            first_pos = int(flags.argmax())
            t_trip = float(sub.iloc[first_pos]["t_start_sec"]) / 3600.0
            total_h = float(sub["t_start_sec"].iloc[-1]) / 3600.0
            print(f"    Train{tr}: is_degrading=1 from t={t_trip:.1f}h "
                  f"({n_deg}/{int(m.sum())} rows, total {total_h:.1f}h)")
        else:
            print(f"    Train{tr}: is_degrading never trips ({int(m.sum())} rows)")

    # Time / index / id columns are NEVER used as features — RUL must be
    # learned from sensor signals only, not from "how far we are".
    drop_cols = [
        "train_id", "file_idx", "file_name",
        "t_start_sec", "time_to_eol_sec", "life_frac",
        "RUL",
    ]
    feature_cols = [c for c in df_all.columns if c not in drop_cols]
    print(f"  feature count: {len(feature_cols)}  (excluded: {drop_cols})")

    if args.single_train:
        # Temporal split: chronological 70/30 within the single Train
        n = len(df_all)
        cut = int(n * 0.7)
        train_mask = np.zeros(n, dtype=bool); train_mask[:cut] = True
        valid_mask = ~train_mask
        valid_label = f"train{args.single_train}_holdout"
        suffix = f"_train{args.single_train}"
    else:
        train_mask = df_all.train_id.isin(TRAIN_IDS).to_numpy()
        valid_mask = df_all.train_id.isin(VALID_IDS).to_numpy()
        valid_label = f"train{VALID_IDS[0]}"
        suffix = ""

    X_train = df_all.loc[train_mask, feature_cols]
    y_train = df_all.loc[train_mask, "RUL"]
    X_valid = df_all.loc[valid_mask, feature_cols]
    y_valid = df_all.loc[valid_mask, "RUL"]
    print(f"  train rows: {len(X_train)}  valid rows: {len(X_valid)}  ({valid_label})")

    print("[5/6] training LightGBM (asymmetric loss + phase weights)...")
    train_ids = df_all.loc[train_mask, "train_id"].to_numpy()
    train_life = df_all.loc[train_mask, "life_frac"].to_numpy()
    sample_weight = compute_sample_weights(y_train.to_numpy(), train_ids, train_life)
    n_decline = int(((sample_weight >= DECLINE_WEIGHT) & (sample_weight < TRANSITION_WEIGHT)).sum())
    n_trans = int((sample_weight >= TRANSITION_WEIGHT).sum())
    print(f"  decline rows: {n_decline}  transition rows: {n_trans}/{len(sample_weight)}")
    print(f"  weights: decline={DECLINE_WEIGHT}, transition={TRANSITION_WEIGHT}, "
          f"life_frac∈{TRANSITION_LIFE_FRAC}")
    print(f"  asymmetric loss: ASYM_ALPHA={ASYM_ALPHA} (overestimation penalty)")

    lgb_params = dict(
        objective=asymmetric_mse_objective,  # custom asymmetric MSE
        metric="rmse",                        # eval metric stays standard
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        feature_fraction=0.8,
        random_state=42,
        verbose=-1,
    )
    model = lgb.LGBMRegressor(**lgb_params, n_estimators=2000)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
    )

    print("[6/6] evaluating + plotting...")
    y_pred_raw = model.predict(X_valid)

    # Pipeline: Raw → Savitzky-Golay smoothing → monotone min-accumulate.
    # Smoothing first absorbs noise so the cummin step doesn't get trapped at
    # a transient dip; then monotone enforces RUL-can-only-decrease physics.
    valid_train_ids = df_all.loc[valid_mask, "train_id"].to_numpy()
    y_pred_smooth = y_pred_raw.copy()
    y_pred = y_pred_raw.copy()
    for tr in np.unique(valid_train_ids):
        idx = np.where(valid_train_ids == tr)[0]
        seg = y_pred_raw[idx]
        # window must be odd and ≤ len(seg)
        w = min(SAVGOL_WINDOW, len(seg) // 2 * 2 - 1)
        if w >= SAVGOL_POLY + 2:
            seg_s = savgol_filter(seg, window_length=w, polyorder=SAVGOL_POLY)
        else:
            seg_s = seg
        y_pred_smooth[idx] = seg_s
        y_pred[idx] = np.minimum.accumulate(seg_s)

    mae_raw = mean_absolute_error(y_valid, y_pred_raw)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse_raw = float(np.sqrt(mean_squared_error(y_valid, y_pred_raw)))
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))

    # Over/under-prediction split (monotone) — verifies asymmetric loss is biting
    err = y_pred - y_valid.to_numpy()
    over = err[err > 0]; under = -err[err < 0]
    print(f"  Validation MAE  : raw={mae_raw:.3f}h  monotone={mae:.3f}h")
    print(f"  Validation RMSE : raw={rmse_raw:.3f}h  monotone={rmse:.3f}h")
    print(f"  over-prediction : n={len(over):3d}  mean={over.mean() if len(over) else 0:.2f}h  max={over.max() if len(over) else 0:.2f}h")
    print(f"  under-prediction: n={len(under):3d}  mean={under.mean() if len(under) else 0:.2f}h  max={under.max() if len(under) else 0:.2f}h")
    print(f"  best_iter       : {model.best_iteration_}")

    fig, ax = plt.subplots(figsize=(12, 5))
    t_h_train = df_all.loc[train_mask, "t_start_sec"].to_numpy() / 3600.0
    t_h_valid = df_all.loc[valid_mask, "t_start_sec"].to_numpy() / 3600.0
    if args.single_train:
        # Show train portion as well so the split is visible
        y_train_pred = model.predict(X_train)
        ax.plot(t_h_train, y_train.to_numpy(), color="tab:blue", lw=1.4, label="Actual (train)")
        ax.plot(t_h_train, y_train_pred, color="tab:cyan", lw=0.9, alpha=0.7, label="Pred (train)")
        ax.axvline(t_h_valid[0], color="k", ls="--", lw=0.8, alpha=0.6, label="70/30 split")
    ax.plot(t_h_valid, y_valid.to_numpy(), label="Actual (valid)", color="tab:orange", lw=1.6)
    ax.plot(t_h_valid, y_pred_raw,    label="Pred raw",       color="lightgray",  lw=1.0, alpha=0.9)
    ax.plot(t_h_valid, y_pred_smooth, label="Pred savgol",    color="tab:cyan",   lw=1.0, alpha=0.85)
    ax.plot(t_h_valid, y_pred,        label="Pred mono",      color="tab:red",    lw=1.4)
    ax.set_title(f"RUL Prediction — {valid_label}  "
                 f"MAE raw={mae_raw:.2f}h → mono={mae:.2f}h  "
                 f"RMSE raw={rmse_raw:.2f}h → mono={rmse:.2f}h")
    ax.set_xlabel("Time [hours]"); ax.set_ylabel("RUL [hours]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    out_pred = FIG_DIR / f"rul_pred_{valid_label}.png"
    fig.tight_layout(); fig.savefig(out_pred, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_pred}")

    fig, ax = plt.subplots(figsize=(10, 7))
    lgb.plot_importance(model, ax=ax, max_num_features=20,
                        title=f"Top 20 Feature Importance ({valid_label})")
    out_imp = FIG_DIR / f"rul_importance{suffix}.png"
    fig.tight_layout(); fig.savefig(out_imp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_imp}")


if __name__ == "__main__":
    main()
