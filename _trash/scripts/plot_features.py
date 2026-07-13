"""Feature dashboard for outputs/features_utils/train{N}.parquet.

Single multi-panel figure per Train showing every column in the parquet:
  Row 1: Operation overlay (RPM, Torque, TC SP Front/Rear)
  Row 2: Raw time-domain — RMS / Kurt / CF (4 channels overlaid)
  Row 3: Filter-band time-domain — RMS_filter / Kurt_filter / CF_filter
  Row 4: Envelope band heatmaps — one per channel (9 bands × N files)

Pre-processing applied at viz time (parquet untouched):
  - RPM regime split (threshold 850 RPM): each file's envelope baseline is
    drawn from early-life files in its own regime — 700 RPM files compared
    to 700 RPM baseline, 950 RPM to 950 RPM baseline.
  - Outlier clipping:
      * Kurtosis (raw + filter) hard-capped at 10
      * RMS / band_filter_rms / envelope bands capped at 99th percentile
    Skew/CF are left untouched (not requested).

Usage:
  python scripts/plot_features.py 3              # Train3 → outputs/figures/features_train3.png
  python scripts/plot_features.py --all          # all 4 Trains
  python scripts/plot_features.py 3 --show       # display interactively
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
FEAT_DIR = ROOT / "outputs" / "features_utils"
FIG_DIR = ROOT / "outputs" / "figures"

CHANNELS = ("CH1", "CH2", "CH3", "CH4")
CH_COLORS = {"CH1": "tab:blue", "CH2": "tab:orange",
             "CH3": "tab:green", "CH4": "tab:red"}
ENV_BANDS = (
    "BPFI_1x", "BPFI_2x", "BPFI_3x",
    "BPFO_1x", "BPFO_2x", "BPFO_3x",
    "BSF_1x",  "BSF_2x",  "BSF_3x",
)

RPM_REGIME_THRESHOLD = 850.0  # Hz; below = "low" (~700), above = "high" (~950)
KURT_CAP = 10.0
PCTILE_CAP = 99.0


def assign_rpm_regime(df: pd.DataFrame,
                      threshold: float = RPM_REGIME_THRESHOLD) -> np.ndarray:
    """Label each file's RPM regime — 'low' (~700) or 'high' (~950)."""
    return np.where(df["rpm_mean"].to_numpy() < threshold, "low", "high")


def clip_features(df: pd.DataFrame,
                  kurt_cap: float = KURT_CAP,
                  pctile_cap: float = PCTILE_CAP) -> pd.DataFrame:
    """Cap outliers per column. Returns a new DataFrame; leaves df unchanged.

    - Hard cap: Kurt / Kurt_filter at kurt_cap
    - Percentile cap (upper): RMS, band_filter_rms, RMS_filter, env_*
    Skew / CF columns left untouched.
    """
    out = df.copy()
    for ch in CHANNELS:
        for suffix in ("kurt", "kurt_filter"):
            col = f"{ch}_{suffix}"
            out[col] = out[col].clip(upper=kurt_cap)
        pct_cols = [f"{ch}_rms", f"{ch}_rms_filter", f"{ch}_band_filter_rms"]
        pct_cols += [f"{ch}_env_{b}" for b in ENV_BANDS]
        for col in pct_cols:
            cap = float(np.nanpercentile(out[col], pctile_cap))
            out[col] = out[col].clip(upper=cap)
    return out


def _line_panel(ax, t_h, df, suffix, title, ylabel, log=False):
    for ch in CHANNELS:
        ax.plot(t_h, df[f"{ch}_{suffix}"], color=CH_COLORS[ch],
                lw=1.0, label=ch)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(alpha=0.3)


def _env_heatmap(ax, df, ch, t_h, regime):
    """9 envelope bands × N_files heatmap, log-normalized per RPM regime.

    For each band:
        baseline_low  = median over first 10% of low-RPM files
        baseline_high = median over first 10% of high-RPM files
    Each file is then divided by its own regime's baseline before log10.
    """
    M = np.array([df[f"{ch}_env_{b}"].to_numpy() for b in ENV_BANDS])
    base_per_file = np.empty_like(M)
    life = df["life_frac"].to_numpy()
    for r in ("low", "high"):
        in_regime = regime == r
        if not in_regime.any():
            continue
        # Early-life within this regime = first 10% of *life*, intersected with regime
        early = in_regime & (life < 0.10)
        if not early.any():
            # Fallback: take the earliest 10% of files that are in this regime
            idxs = np.where(in_regime)[0]
            n_keep = max(1, len(idxs) // 10)
            early_mask = np.zeros_like(in_regime)
            early_mask[idxs[:n_keep]] = True
            early = early_mask
        base_vec = np.median(M[:, early], axis=1, keepdims=True)
        base_vec[base_vec == 0] = np.nan
        base_per_file[:, in_regime] = base_vec  # broadcast across files in regime

    M_norm = np.log10(M / base_per_file + 1e-6)
    im = ax.imshow(M_norm, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, origin="lower",
                   extent=[t_h[0], t_h[-1], -0.5, len(ENV_BANDS) - 0.5])
    ax.set_yticks(range(len(ENV_BANDS)))
    ax.set_yticklabels(ENV_BANDS, fontsize=8)
    ax.set_title(f"{ch} envelope bands", fontsize=10)
    ax.set_xlabel("Time [hours]")
    return im


def plot_train(train_id: int, save: bool = True, show: bool = False) -> Path | None:
    df_raw = pd.read_parquet(FEAT_DIR / f"train{train_id}.parquet")
    df_raw = df_raw.sort_values("file_idx").reset_index(drop=True)
    df = clip_features(df_raw)
    regime = assign_rpm_regime(df)
    t_h = df["t_start_sec"].to_numpy() / 3600.0
    n_low = int((regime == "low").sum())
    n_high = int((regime == "high").sum())

    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(4, 4, figure=fig,
                  height_ratios=[1.0, 1.0, 1.0, 1.6],
                  hspace=0.45, wspace=0.30)

    # ---------- Row 1: Operation ----------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_h, df["rpm_mean"], color="tab:blue", lw=1.0)
    ax.fill_between(t_h, df["rpm_mean"] - df["rpm_std"],
                    df["rpm_mean"] + df["rpm_std"],
                    color="tab:blue", alpha=0.2)
    ax.set_title("RPM (mean ± std)", fontsize=10)
    ax.set_ylabel("rpm"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_h, df["torque_mean"], color="tab:purple", lw=1.0, label="mean")
    ax.plot(t_h, df["torque_min"], color="tab:red", lw=0.7, alpha=0.7, label="min")
    ax.axhline(-20, color="red", ls="--", lw=0.7, alpha=0.5)
    ax.set_title("Torque [Nm]", fontsize=10)
    ax.set_ylabel("Nm"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t_h, df["tcf_max"], color="tab:orange", lw=1.0, label="Front max")
    ax.plot(t_h, df["tcr_max"], color="tab:green", lw=1.0, label="Rear max")
    ax.axhline(200, color="red", ls="--", lw=0.7, alpha=0.5)
    ax.set_title("TC SP Front/Rear [°C]", fontsize=10)
    ax.set_ylabel("°C"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # life_frac sanity bar
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(t_h, df["life_frac"], color="k", lw=1.0)
    ax.set_title("life_frac (0 → 1)", fontsize=10)
    ax.set_ylabel("frac"); ax.grid(alpha=0.3)

    # ---------- Row 2: Raw time-domain ----------
    _line_panel(fig.add_subplot(gs[1, 0]), t_h, df, "rms",
                "Raw RMS", "RMS")
    _line_panel(fig.add_subplot(gs[1, 1]), t_h, df, "kurt",
                "Raw Kurtosis", "Kurt", log=True)
    _line_panel(fig.add_subplot(gs[1, 2]), t_h, df, "cf",
                "Raw Crest Factor", "CF")
    _line_panel(fig.add_subplot(gs[1, 3]), t_h, df, "band_filter_rms",
                "Band-filter RMS (BP energy)", "RMS")

    # Single shared legend on row 2 last panel
    fig.axes[-1].legend(fontsize=8, ncol=4, loc="upper left")

    # ---------- Row 3: Filter-band time-domain ----------
    _line_panel(fig.add_subplot(gs[2, 0]), t_h, df, "rms_filter",
                "Filter RMS (post-BP)", "RMS")
    _line_panel(fig.add_subplot(gs[2, 1]), t_h, df, "kurt_filter",
                "Filter Kurtosis (post-BP)", "Kurt", log=True)
    _line_panel(fig.add_subplot(gs[2, 2]), t_h, df, "cf_filter",
                "Filter Crest Factor (post-BP)", "CF")
    _line_panel(fig.add_subplot(gs[2, 3]), t_h, df, "skew",
                "Raw Skewness", "Skew")

    # ---------- Row 4: Envelope band heatmaps (per channel) ----------
    last_im = None
    for j, ch in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[3, j])
        last_im = _env_heatmap(ax, df, ch, t_h, regime)
    cbar = fig.colorbar(last_im, ax=fig.axes[-4:], orientation="horizontal",
                        shrink=0.5, pad=0.18, fraction=0.04, aspect=40)
    cbar.set_label(
        f"log10(env band RMS / early-life baseline of same RPM regime)  "
        f"— red = grew, blue = shrank", fontsize=9)

    n_files = len(df)
    fig.suptitle(
        f"Train{train_id} — feature dashboard  "
        f"({n_files} files, {t_h[-1]:.1f} h)  |  "
        f"RPM regimes: low={n_low}, high={n_high}  |  "
        f"Kurt cap={KURT_CAP:.0f}, percentile cap={PCTILE_CAP:.0f}%",
        fontsize=12, y=0.995)

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / f"features_train{train_id}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return FIG_DIR / f"features_train{train_id}.png" if save else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train_id", nargs="?", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--all", action="store_true", help="plot all 4 Trains")
    p.add_argument("--show", action="store_true", help="display interactively")
    args = p.parse_args()

    if args.all:
        for tr in (1, 2, 3, 4):
            plot_train(tr, save=True, show=False)
    elif args.train_id is not None:
        plot_train(args.train_id, save=True, show=args.show)
    else:
        p.error("provide train_id (1-4) or --all")


if __name__ == "__main__":
    main()
