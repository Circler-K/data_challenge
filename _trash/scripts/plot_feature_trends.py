"""Generate per-feature trend plots: 4 trains × 4 channels.

Layout (matches user's HRMS reference):
  outputs/feature_trends/{feature_name}/Train{1..4}.png
  Each PNG: 2x2 grid of CH1, CH2, CH3, CH4 over file_idx (time order).
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "outputs" / "features_full"
OUT_DIR = ROOT / "outputs" / "feature_trends"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> dict[int, pd.DataFrame]:
    out = {}
    for tr in [1, 2, 3, 4]:
        p = FEAT_DIR / f"train{tr}.parquet"
        if p.exists():
            out[tr] = pd.read_parquet(p).sort_values("file_idx").reset_index(drop=True)
    return out


def discover_features(dfs: dict[int, pd.DataFrame]) -> list[str]:
    """Find all features that exist as CH1_<name>, CH2_<name>, CH3_<name>, CH4_<name>."""
    cols = set.intersection(*(set(df.columns) for df in dfs.values()))
    feats = set()
    for c in cols:
        if c.startswith("CH1_"):
            name = c[4:]
            if all(f"CH{i}_{name}" in cols for i in (1, 2, 3, 4)):
                feats.add(name)
    return sorted(feats)


def plot_feature(feat_name: str, dfs: dict[int, pd.DataFrame]):
    out_dir = OUT_DIR / feat_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for tr, df in dfs.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        x = df["file_idx"].to_numpy()
        for i, ax in enumerate(axes.ravel()):
            ch = f"CH{i+1}"
            col = f"{ch}_{feat_name}"
            if col not in df.columns:
                ax.set_visible(False)
                continue
            y = df[col].to_numpy()
            ax.plot(x, y, marker="o", ms=3, lw=0.8, color="steelblue")
            ax.set_title(f"Train{tr} - {ch}  {feat_name}", fontsize=11)
            ax.set_xlabel("file index (time order)")
            ax.set_ylabel(feat_name)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = out_dir / f"Train{tr}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)


def trend_score(y: np.ndarray) -> float:
    """Spearman correlation with time index — proxy for monotonicity."""
    if len(y) < 5 or np.all(y == y[0]):
        return 0.0
    from scipy.stats import spearmanr
    rho, _ = spearmanr(np.arange(len(y)), y)
    return float(rho) if np.isfinite(rho) else 0.0


def make_summary(dfs: dict[int, pd.DataFrame], feats: list[str]) -> pd.DataFrame:
    """Per (feature × train × channel) Spearman ρ vs time. Higher = more upward trend."""
    rows = []
    for feat in feats:
        for tr, df in dfs.items():
            for i in range(4):
                col = f"CH{i+1}_{feat}"
                if col not in df.columns:
                    continue
                rho = trend_score(df[col].to_numpy())
                rows.append(dict(feature=feat, train=tr, ch=f"CH{i+1}", rho=rho))
    return pd.DataFrame(rows)


def main():
    dfs = load_all()
    if not dfs:
        print("No parquet files in", FEAT_DIR)
        return
    print(f"loaded trains: {sorted(dfs.keys())}")
    feats = discover_features(dfs)
    print(f"discovered {len(feats)} per-channel features")

    t0 = time.time()
    for k, feat in enumerate(feats, start=1):
        plot_feature(feat, dfs)
        if k % 5 == 0 or k == len(feats):
            print(f"  [{k:>3}/{len(feats)}] {feat}  ({time.time()-t0:.1f}s)")

    # Summary CSV with monotonicity score
    summ = make_summary(dfs, feats)
    summ_path = OUT_DIR / "_monotonicity_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"summary -> {summ_path}")

    # Top features (averaged |rho| over all trains × channels)
    grand = (summ.assign(abs_rho=lambda d: d["rho"].abs())
                  .groupby("feature")["abs_rho"].mean()
                  .sort_values(ascending=False))
    print("\nTop 15 features by mean |Spearman ρ| across all trains×channels:")
    print(grand.head(15).to_string())


if __name__ == "__main__":
    main()
