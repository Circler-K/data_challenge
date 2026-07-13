"""Fault-frequency physics RUL experiment for KSPHM-KIMM 2026.

Geometry (from _challenge_text.txt): Bearing 30306 tapered roller.
Fault freqs @1000 RPM: BPFI=140Hz, BPFO=93Hz, BSF=78Hz, Cage=6.7Hz.
As orders (/16.667 rev/s): BPFI=8.40, BPFO=5.58, BSF=4.68, FTF=0.402.

Fault-order energy caches: outputs/ot_features/faultorder/{Train*,test/Test*}.csv
Columns per channel Ch{0..3}: OT_RMS, Order_Band, Env_Band, Fault_BPFO, Fault_BPFI, Fault_BSF.
(log-scale band energies; healthy baseline ~3.0, failure ~4.3 on rear channels.)

Method: build a Health Indicator (HI) from fault-order energy growth vs a
healthy baseline, normalise to a degradation fraction lf in [0,1] using the
median observed end-of-life (EOL) HI across training bearings, then map
lf -> RUL = CAP - lf*(CAP-FLOOR). Calibrate CAP/FLOOR/channel-agg by LOO.

Compares fault-order HI vs generic broadband (Order_Band) HI:
 - EOL transferability: coefficient of variation (CV) of EOL HI across bearings.
 - LOO A_RUL at cuts 40/50/60.

Analysis only; writes nothing except under outputs/scratch/faultfreq/.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring import a_rul_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FO = os.path.join(ROOT, "outputs", "ot_features", "faultorder")
OUT = os.path.join(ROOT, "outputs", "scratch", "faultfreq")
os.makedirs(OUT, exist_ok=True)

FILE_PERIOD = 600.0
TAIL = 60.0  # last file is +60s of acquisition

TRAIN = {1: 126, 2: 114, 3: 89, 4: 137}
CHANS = [0, 1, 2, 3]
FAULT_COLS = ["Fault_BPFO", "Fault_BPFI", "Fault_BSF"]


def load(name, folder=FO):
    return pd.read_csv(os.path.join(folder, name + ".csv"))


def total_life(n):
    return (n - 1) * FILE_PERIOD + TAIL


def actual_rul(n, cut):
    # RUL at file index `cut` (1-based): time from cut to failure.
    return (n - cut) * FILE_PERIOD + TAIL


def baseline(df, k=10):
    """Healthy baseline = mean of first k files for each column."""
    return df.iloc[:k].mean(numeric_only=True)


def hi_series(df, cols, base, agg):
    """HI = aggregated positive growth of `cols` over baseline.
    cols: list of column names. agg: 'mean','max', or a single channel index list."""
    growth = []
    for c in cols:
        g = (df[c] - base[c]).clip(lower=0)
        growth.append(g.values)
    growth = np.vstack(growth)  # (ncols, nfiles)
    if agg == "mean":
        return growth.mean(axis=0)
    if agg == "max":
        return growth.max(axis=0)
    raise ValueError(agg)


def fault_cols_for(channels):
    return [f"Ch{ch}_{fc}" for ch in channels for fc in FAULT_COLS]


def generic_cols_for(channels):
    return [f"Ch{ch}_Order_Band" for ch in channels]


def smooth(x, w=5):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


# ---------- EOL transferability (CV) ----------
def eol_his(cols_fn, channels, agg):
    """End-of-life HI for each training bearing (value at last file)."""
    eols = {}
    for b, n in TRAIN.items():
        df = load(f"Train{b}")
        base = baseline(df)
        hi = smooth(hi_series(df, cols_fn(channels), base, agg))
        eols[b] = hi[-1]
    return eols


# ---------- LOO calibration ----------
def predict_rul(hi_cut, eol_ref, cap, floor):
    """Map current HI to RUL via degradation fraction lf = hi_cut/eol_ref."""
    lf = np.clip(hi_cut / eol_ref if eol_ref > 0 else 0.0, 0.0, 1.0)
    return cap - lf * (cap - floor)


def loo_eval(cols_fn, channels, agg, cap, floor, cuts=(40, 50, 60)):
    """For each cut, LOO over 4 bearings: calibrate eol_ref on other 3, predict held-out."""
    results = {}
    for cut in cuts:
        scores, preds, acts = [], [], []
        for held in TRAIN:
            others = [b for b in TRAIN if b != held]
            eol_ref = np.median([
                smooth(hi_series(load(f"Train{b}"), cols_fn(channels),
                                 baseline(load(f"Train{b}")), agg))[-1]
                for b in others
            ])
            df = load(f"Train{held}")
            base = baseline(df)
            hi = smooth(hi_series(df, cols_fn(channels), base, agg))
            hi_cut = hi[cut - 1]
            pred = predict_rul(hi_cut, eol_ref, cap, floor)
            act = actual_rul(TRAIN[held], cut)
            scores.append(float(a_rul_score(act, pred)))
            preds.append(pred)
            acts.append(act)
        results[cut] = (np.mean(scores), preds, acts)
    return results


def main():
    print("=" * 70)
    print("FAULT FREQUENCIES (from _challenge_text.txt, @1000 RPM):")
    print("  BPFI=140Hz BPFO=93Hz BSF=78Hz Cage=6.7Hz | rev/s=16.667")
    for nm, hz in [("BPFI", 140), ("BPFO", 93), ("BSF", 78), ("FTF", 6.7)]:
        print(f"  {nm} order = {hz/16.6667:.3f}")
    print("=" * 70)

    # ---- EOL transferability: fault-order vs generic, per channel-set ----
    print("\n[EOL HI transferability] CV across 4 training bearings (lower=better)")
    channel_sets = {
        "all4": [0, 1, 2, 3],
        "rear(2,3)": [2, 3],
        "Ch3": [3],
        "Ch2": [2],
    }
    cv_table = []
    for label, chs in channel_sets.items():
        for agg in ["mean", "max"]:
            fo = eol_his(fault_cols_for, chs, agg)
            ge = eol_his(generic_cols_for, chs, agg)
            fo_v = np.array(list(fo.values()))
            ge_v = np.array(list(ge.values()))
            cv_fo = fo_v.std() / fo_v.mean()
            cv_ge = ge_v.std() / ge_v.mean()
            cv_table.append((label, agg, cv_fo, cv_ge, fo, ge))
            print(f"  ch={label:10s} agg={agg:4s} | faultorder CV={cv_fo:.3f} "
                  f"generic CV={cv_ge:.3f} | EOL_fo={[round(v,2) for v in fo_v]}")

    # ---- Pick best config by LOO (grid over channel-set, agg, cap, floor) ----
    print("\n[LOO grid search] fault-order HI")
    caps = [55000, 60000, 65000, 70000]
    floors = [1200, 1800, 3000]
    best = None
    for label, chs in channel_sets.items():
        for agg in ["mean", "max"]:
            for cap in caps:
                for floor in floors:
                    res = loo_eval(fault_cols_for, chs, agg, cap, floor)
                    avg = np.mean([res[c][0] for c in (40, 50, 60)])
                    cut50 = res[50][0]
                    key = (avg, cut50, label, agg, cap, floor)
                    if best is None or avg > best[0]:
                        best = (avg, cut50, label, agg, cap, floor, res)
    avg, cut50, label, agg, cap, floor, res = best
    print(f"  BEST fault-order: ch={label} agg={agg} cap={cap} floor={floor}")
    print(f"    LOO cut50={cut50:.4f}  avg(40,50,60)={avg:.4f}")
    for c in (40, 50, 60):
        print(f"    cut{c}: score={res[c][0]:.4f} preds={[int(p) for p in res[c][1]]} "
              f"acts={[int(a) for a in res[c][2]]}")

    # ---- Same grid for generic, to compare ----
    print("\n[LOO grid search] generic Order_Band HI")
    bestg = None
    for label_g, chs_g in channel_sets.items():
        for agg_g in ["mean", "max"]:
            for cap_g in caps:
                for floor_g in floors:
                    res2 = loo_eval(generic_cols_for, chs_g, agg_g, cap_g, floor_g)
                    avg2 = np.mean([res2[c][0] for c in (40, 50, 60)])
                    if bestg is None or avg2 > bestg[0]:
                        bestg = (avg2, res2[50][0], label_g, agg_g, cap_g, floor_g, res2)
    avg2, cut50g, lg, ag, capg, floorg, resg = bestg
    print(f"  BEST generic: ch={lg} agg={ag} cap={capg} floor={floorg}")
    print(f"    LOO cut50={cut50g:.4f}  avg(40,50,60)={avg2:.4f}")

    # ---- Final validation predictions using BEST fault-order config ----
    print("\n[VALIDATION PREDICTIONS] using best fault-order config")
    print(f"  config: ch={label} agg={agg} cap={cap} floor={floor}")
    chs = channel_sets[label]
    eol_ref = np.median([
        smooth(hi_series(load(f"Train{b}"), fault_cols_for(chs),
                         baseline(load(f"Train{b}")), agg))[-1]
        for b in TRAIN
    ])
    print(f"  eol_ref (median train EOL HI)={eol_ref:.3f}  cap={cap} floor={floor}")
    val_preds = {}
    for t in range(1, 7):
        df = load(f"Test{t}", os.path.join(FO, "test"))
        base = baseline(df)
        hi = smooth(hi_series(df, fault_cols_for(chs), base, agg))
        hi_cut = hi[-1]  # file 50
        pred = predict_rul(hi_cut, eol_ref, cap, floor)
        val_preds[t] = pred
        lf = np.clip(hi_cut / eol_ref, 0, 1)
        print(f"  Val{t}=Test{t}: HI@50={hi_cut:.3f} lf={lf:.3f} -> RUL={int(round(pred))} s")

    # Diagnostic: HI@50 vs train EOL HI distribution
    train_eol = [smooth(hi_series(load(f"Train{b}"), fault_cols_for(chs),
                 baseline(load(f"Train{b}")), agg))[-1] for b in TRAIN]
    train_hi50 = [smooth(hi_series(load(f"Train{b}"), fault_cols_for(chs),
                  baseline(load(f"Train{b}")), agg))[49] for b in TRAIN]
    print(f"\n  [diag] train EOL HI = {[round(v,3) for v in train_eol]}")
    print(f"  [diag] train HI@file50 = {[round(v,3) for v in train_hi50]} "
          f"(true RUL@50 = {[int(actual_rul(TRAIN[b],50)) for b in TRAIN]})")

    # Anchor scoring against known V2 ground truth filter (~72906)
    print("\n  raw Val2 prediction =", int(round(val_preds[2])), " (ground-truth ~72906)")

    # Score against provided anchor as sanity (not real GT for others)
    pred_vec = [val_preds[i] for i in range(1, 7)]
    print("  6-pred vector:", [int(round(p)) for p in pred_vec])

    # ---- Generic-model validation preds for direct comparison ----
    print("\n[VALIDATION PREDICTIONS] best GENERIC config (for comparison)")
    print(f"  config: ch={lg} agg={ag} cap={capg} floor={floorg}")
    chs_g = channel_sets[lg]
    eol_ref_g = np.median([
        smooth(hi_series(load(f"Train{b}"), generic_cols_for(chs_g),
                         baseline(load(f"Train{b}")), ag))[-1]
        for b in TRAIN])
    gvec = []
    for t in range(1, 7):
        df = load(f"Test{t}", os.path.join(FO, "test"))
        hi = smooth(hi_series(df, generic_cols_for(chs_g), baseline(df), ag))
        gvec.append(predict_rul(hi[-1], eol_ref_g, capg, floorg))
    print("  generic 6-pred:", [int(round(p)) for p in gvec], "rawVal2=", int(round(gvec[1])))


if __name__ == "__main__":
    main()
