"""Option B — stage-based RUL for the 6 validation bearings.

Rationale (see analysis): bearing HI is flat for the first ~70% of life then
knees up. So we estimate each validation bearing's degradation STAGE by matching
its HI level to where the training bearings reach that HI (life fraction), then
RUL = (1 - life_fraction) * typical_total_life. This yields DISTINCT, monotone,
stage-grounded predictions (no degenerate clustering) and is interpretable for
the technical report.

NOTE: self-validation (LOTO) cannot rank this against the conservative Wiener
model — the asymmetric metric is saturated (~0.42 for everything incl. a
constant). The preliminary submission is the calibration signal.

Run:  python scripts/predict_stage_rul.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import wiener_rul as W  # noqa: E402
from predict_test_wiener import (global_baseline, hi_global,  # noqa: E402
                                 extract_bearing, BEARINGS, TEAM)

FLOOR_SEC = 1800.0       # validation cut before failure -> RUL >= ~0.5 h
PCT = 50                 # life-fraction percentile across training curves
T_STAT = "median"        # typical total life used to scale RUL


def life_curves(train: dict, mu, sigma):
    out = []
    for df in train.values():
        h = hi_global(df, mu, sigma)
        t = df["t_sec"].to_numpy()
        out.append((h, t / (t[-1] + 60)))
    return out


def stage_rul(q_hi: float, curves, total_life: float, pct: float = PCT) -> float:
    """life_frac where training HI first reaches q -> RUL = (1-lf)*total_life."""
    lfs = []
    for h, lf in curves:
        idx = np.where(h >= q_hi)[0]
        lfs.append(lf[idx[0]] if len(idx) else 1.05)   # >1 => already past EOL level
    lf = float(np.percentile(lfs, pct))
    return max((1.0 - min(lf, 1.0)) * total_life, FLOOR_SEC)


def main():
    train = W.load("est")
    mu, sigma = global_baseline(train)
    curves = life_curves(train, mu, sigma)
    lives = np.array([df["t_sec"].iloc[-1] + 60 for df in train.values()])
    T = float(np.median(lives) if T_STAT == "median" else np.percentile(lives, 25))
    print(f"typical total life T({T_STAT}) = {T/3600:.1f}h\n")

    out = []
    for name in BEARINGS:
        df = extract_bearing(name)
        hi = hi_global(df, mu, sigma)
        rul = stage_rul(float(hi[-1]), curves, T)
        out.append(dict(File=name, hi_last=round(float(hi[-1]), 2),
                        RUL_Score=int(round(rul)), rul_h=round(rul / 3600, 2)))
    res = pd.DataFrame(out)
    print(res.to_string(index=False))

    HERE = Path(__file__).resolve().parent   # 결과는 이 폴더 안에만 쓴다(라이브 제출파일 미수정)
    sub = res[["File", "RUL_Score"]]
    sub.to_excel(HERE / f"{TEAM}_validation.xlsx",
                 index=False, sheet_name="Sheet1")
    res.to_csv(HERE / "test_rul_stage.csv", index=False)
    print(f"\nsaved {HERE / (TEAM + '_validation.xlsx')} (Option B, stage-based)")


if __name__ == "__main__":
    main()
