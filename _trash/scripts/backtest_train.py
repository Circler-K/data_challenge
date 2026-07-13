"""Back-test the FINAL submission method (predict_final.py, RMS degradation-fraction)
on the TRAIN bearings, whose true RUL is known (they ran to failure).

Two modes:
  in-sample : heal/eol anchors + CAP from ALL 4 train bearings (the real submission's
              anchors). The held bearing helps define its own anchors -> optimistic.
  LOO       : heal/eol/CAP from the OTHER 3 bearings only -> honest generalization.

Evaluated at cut=50 (same file index the validation/test bearings are cut at), and
also the trajectory across cuts 30/40/50/60/70 to see how the method tracks decay.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import method_decision as M  # noqa: E402
from src.scoring import a_rul_score  # noqa: E402

FLOOR_SUB = 3600.0  # submission floor
TRAINS = [1, 2, 3, 4]


def max_rms_series(df):
    cols = [c for c in df.columns if c.endswith("OT_RMS")]
    return df[cols].to_numpy(float).max(1)


def anchors(train, names):
    heal = float(np.median([np.median(max_rms_series(train[t])[:max(3, int(len(train[t]) * 0.15))])
                            for t in names]))
    eol = float(np.median([max_rms_series(train[t])[-1] for t in names]))
    cap = float(max(M.actual_rul(t, 50) for t in names))  # max RUL@file50 across bearings
    return heal, eol, cap


def predict(r, heal, eol, cap, floor):
    lf = min(max((r - heal) / (eol - heal), 0.0), 1.0)
    return cap - lf * (cap - floor)


def main():
    train = M.load(M.FO, TRAINS, "Train")

    # ---- cut=50 back-test ----
    print("=== TRAIN back-test @ cut=50 (RMS degradation-fraction = final method) ===\n")
    for mode in ("in-sample", "LOO"):
        rows, acts, preds = [], [], []
        # in-sample anchors use all 4 + the real submission CAP/FLOOR
        if mode == "in-sample":
            heal, eol, cap = anchors(train, TRAINS)
            cap = 53153.0  # the literal CAP in predict_final.py
        for tr in TRAINS:
            if mode == "LOO":
                others = [t for t in TRAINS if t != tr]
                heal, eol, cap = anchors(train, others)
            r = float(max_rms_series(train[tr])[49])  # file index 50
            pred = predict(r, heal, eol, cap, FLOOR_SUB)
            act = float(M.actual_rul(tr, 50))
            sc = float(a_rul_score(act, pred))
            err = 100 * (act - pred) / act
            rows.append((tr, r, act, pred, err, sc))
            acts.append(act); preds.append(pred)
        print(f"-- {mode}  (heal={heal:.3f} eol={eol:.3f} cap={cap:.0f}) --")
        print(f"{'Train':>6} {'RMS50':>7} {'actRUL':>8} {'predRUL':>8} {'Er%':>7} {'A_RUL':>6}")
        for tr, r, act, pred, err, sc in rows:
            print(f"{tr:>6} {r:>7.3f} {act:>8.0f} {pred:>8.0f} {err:>7.1f} {sc:>6.3f}")
        print(f"{'mean':>6} {'':>7} {'':>8} {'':>8} {'':>7} {np.mean([x[5] for x in rows]):>6.3f}\n")

    # ---- trajectory across cuts ----
    print("=== A_RUL vs cut (LOO anchors) - does the method track decay over time? ===")
    cuts = [30, 40, 50, 60, 70]
    print(f"{'cut':>5} " + " ".join(f"T{t:>5}" for t in TRAINS) + f" {'mean':>7}")
    for cut in cuts:
        line, scs = [], []
        for tr in TRAINS:
            others = [t for t in TRAINS if t != tr]
            heal, eol, cap = anchors(train, others)
            n = len(train[tr])
            if cut > n:
                line.append("  n/a"); continue
            r = float(max_rms_series(train[tr])[cut - 1])
            pred = predict(r, heal, eol, cap, FLOOR_SUB)
            act = float(M.actual_rul(tr, cut))
            sc = float(a_rul_score(act, pred))
            scs.append(sc); line.append(f"{sc:6.3f}")
        print(f"{cut:>5} " + " ".join(line) + f" {np.mean(scs):>7.3f}")


if __name__ == "__main__":
    main()
