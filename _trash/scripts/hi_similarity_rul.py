"""Fix for the Wiener model's degenerate clustering: map ABSOLUTE HI level ->
RUL via similarity to the training bearings' HI-vs-remaining-life curves.

Why: the threshold-FHT model collapsed to 3 discrete outputs (cap / FHT / floor)
because RUL depended on a hard threshold + recent slope and IGNORED the absolute
degradation level — so bearings at different HI levels got identical RUL.

This model instead pools all training (HI_level, RUL_remaining) pairs and, for a
query HI, takes a CONSERVATIVE low percentile of the remaining-life of the
training files at a similar HI level (k-NN in HI space). RUL then varies
smoothly and monotonically with HI level — no clustering.

  - LOTO harness: validates on training (leave-one-bearing-out for both the HI
    baseline AND the similarity library), scored with official A_RUL.
  - predict: applies the all-training library to the 6 validation bearings.

Run:  python scripts/hi_similarity_rul.py            # LOTO score
      python scripts/hi_similarity_rul.py --predict   # validation RUL + xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import wiener_rul as W  # noqa: E402
from src.scoring import a_rul_score, error_pct  # noqa: E402
from predict_test_wiener import (global_baseline, hi_global,  # noqa: E402
                                 extract_bearing, BEARINGS, TEAM)

CUT_FRACS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
K = 40          # neighbours in HI space
PCT = 35        # conservative low percentile of neighbour RULs
FLOOR_SEC = 1800.0


def training_library(train: dict, mu, sigma, exclude: int | None = None):
    """Pool (HI_level, RUL_remaining_sec) over all training files (optionally
    excluding one bearing for LOTO)."""
    his, ruls = [], []
    for tr, df in train.items():
        if tr == exclude:
            continue
        hi = hi_global(df, mu, sigma)
        t = df["t_sec"].to_numpy()
        eol = t[-1] + 60
        his.append(hi)
        ruls.append(eol - t)
    return np.concatenate(his), np.concatenate(ruls)


def rul_from_hi(q_hi: float, lib_hi: np.ndarray, lib_rul: np.ndarray,
                k: int = K, pct: float = PCT) -> float:
    """Conservative similarity RUL: low percentile of the remaining-life of the
    k training files whose HI is closest to the query HI."""
    idx = np.argsort(np.abs(lib_hi - q_hi))[:k]
    return float(max(np.percentile(lib_rul[idx], pct), FLOOR_SEC))


def loto_score(train: dict) -> pd.DataFrame:
    eol_sec = {tr: float(df["t_sec"].iloc[-1] + 60) for tr, df in train.items()}
    rows = []
    for held in train:
        others = [t for t in train if t != held]
        mu, sigma = global_baseline({t: train[t] for t in others})
        lib_hi, lib_rul = training_library(train, mu, sigma, exclude=held)
        df = train[held]
        hi = hi_global(df, mu, sigma)
        t = df["t_sec"].to_numpy()
        n = len(df)
        for fr in CUT_FRACS:
            c = min(n - 1, max(1, int(round(fr * n)) - 1))
            act = eol_sec[held] - t[c]
            pred = rul_from_hi(float(hi[c]), lib_hi, lib_rul)
            rows.append(dict(held=held, cut_frac=fr, act_h=act / 3600,
                             rul_h=pred / 3600, hi_now=float(hi[c]),
                             er=float(error_pct(act, pred)),
                             score=float(a_rul_score(act, pred))))
    return pd.DataFrame(rows)


def predict():
    train = W.load("est")
    mu, sigma = global_baseline(train)
    lib_hi, lib_rul = training_library(train, mu, sigma)
    out = []
    for name in BEARINGS:
        df = extract_bearing(name)            # cached
        hi = hi_global(df, mu, sigma)
        rul = rul_from_hi(float(hi[-1]), lib_hi, lib_rul)
        out.append(dict(bearing=name, hi_last=round(float(hi[-1]), 2),
                        rul_sec=round(rul, 1), rul_h=round(rul / 3600, 3)))
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "outputs" / "test_rul_similarity.csv", index=False)
    sub = pd.DataFrame({"File": res["bearing"],
                        "RUL_Score": res["rul_sec"].round().astype(int)})
    sub.to_excel(ROOT / "outputs" / f"{TEAM}_validation.xlsx", index=False,
                 sheet_name="Sheet1")
    print(res.to_string(index=False))
    print(f"\nsaved outputs/{TEAM}_validation.xlsx + test_rul_similarity.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    args = ap.parse_args()
    if args.predict:
        predict()
        return
    train = W.load("est")
    res = loto_score(train)
    pd.set_option("display.float_format", lambda v: f"{v:7.3f}")
    print(f"=== HI-similarity RUL  (k={K}, pct={PCT}) — LOTO ===")
    print(res.groupby("cut_frac")["score"].mean().to_string())
    print(res.groupby("held")["score"].mean().to_string())
    print(f">>> OVERALL A_RUL = {res.score.mean():.4f}   "
          f"(Wiener-FHT was 0.42; constant-1h 0.42)")


if __name__ == "__main__":
    main()
