"""Extract BOTH feature families in one pass (same estimated RPM -> fair compare):
  GENERAL BAND : Order_Band(1-50), Env_Band(1-50)        [current approach]
  FAULT ORDER  : envelope amplitude at BPFO/BPFI/BSF orders (+harmonics)
                 from the official fault frequencies @1000rpm:
                 BPFO 93Hz->5.58, BPFI 140Hz->8.40, BSF 78Hz->4.68 (shaft=16.667Hz)

Then compare which family yields a better Health Index (monotonicity / trendability
/ late-vs-early discrimination) across the 4 training bearings.

Run (extract one bearing -> CSV):  python scripts/fault_order_features.py --bearing 1
Compare (after all 4 extracted):    python scripts/fault_order_features.py --compare
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.signal import hilbert

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
INHWAN = Path("c:/Users/User/WorkSpace/INHWAN")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(INHWAN))
import rpm_estimator as R                       # noqa: E402
from ot_rpm_impact import order_track, CHS       # noqa: E402
from src.operation import list_vibration_files   # noqa: E402

SAMPLES_PER_REV = 1024
FAULTS = {"BPFO": 5.58, "BPFI": 8.40, "BSF": 4.68}
TOL = 0.15
OUT = ROOT / "outputs" / "ot_features" / "faultorder"

GEN_FEATS = ["Order_Band", "Env_Band"]
FO_FEATS = ["Fault_BPFO", "Fault_BPFI", "Fault_BSF"]


def feats(sig, rpm, ch):
    ot = order_track(np.asarray(sig, float), rpm)
    p = f"Ch{ch}_"
    out = {p + "OT_RMS": float(np.sqrt(np.mean(ot ** 2)))}
    orders = np.fft.rfftfreq(len(ot), d=1.0 / SAMPLES_PER_REV)
    spec = np.abs(np.fft.rfft(ot))
    m = (orders >= 1) & (orders <= 50)
    out[p + "Order_Band"] = float(np.log10(np.sum(spec[m] ** 2) + 1.0))
    env = np.abs(hilbert(ot)); env = env - env.mean()
    es = np.abs(np.fft.rfft(env))
    out[p + "Env_Band"] = float(np.log10(np.sum(es[m] ** 2) + 1.0))
    for fn, fo in FAULTS.items():
        tot = 0.0
        for h in range(1, 4):
            w = (orders >= fo * h - TOL) & (orders <= fo * h + TOL)
            if w.any():
                tot += float(es[w].max())
        out[p + "Fault_" + fn] = float(np.log10(tot + 1.0))
    return out


def extract(tr, test_name=None):
    if test_name is not None:
        from predict_test_wiener import bearing_files
        files = bearing_files(test_name)
    else:
        files = list_vibration_files(tr)
    sig_cache, ch0 = [], []
    for f in files:
        chs = [c[:] for c in TdmsFile.read(str(f)).groups()[0].channels()][:4]
        sig_cache.append(chs); ch0.append(chs[0])
    est = R.refine_stepwise(R.estimate_rpm_series(ch0))
    rows = []
    for k, chs in enumerate(sig_cache):
        row = {"File_Index": k + 1}
        for ch in CHS:
            if ch < len(chs):
                row.update(feats(chs[ch], float(est[k]), ch))
        rows.append(row)
    df = pd.DataFrame(rows); df["t_sec"] = (df["File_Index"] - 1) * 600
    return df


# ---------------- comparison ----------------
def _mono(x):
    d = np.diff(x); return abs((np.sum(d > 0) - np.sum(d < 0)) / max(len(d), 1))


def _trend(x):
    t = np.arange(len(x)); return abs(np.corrcoef(x, t)[0, 1]) if np.std(x) > 0 else 0.0


def build_hi(data, feat_suffixes):
    cols_by = {tr: [f"Ch{c}_{f}" for c in range(4) for f in feat_suffixes
                    if f"Ch{c}_{f}" in data[tr].columns] for tr in data}
    cols = cols_by[next(iter(data))]
    P = np.vstack([data[tr][cols].to_numpy(float)[:max(3, int(len(data[tr]) * 0.15))] for tr in data])
    mu, sd = P.mean(0), P.std(0) + 1e-9
    his = {}
    for tr, df in data.items():
        z = ((df[cols].to_numpy(float) - mu) / sd).mean(1)
        his[tr] = pd.Series(z).rolling(5, min_periods=1).median().cummax().to_numpy()
    return his


def compare():
    data = {tr: pd.read_csv(OUT / f"Train{tr}.csv").sort_values("File_Index").reset_index(drop=True)
            for tr in (1, 2, 3, 4)}
    print("=== HI quality: GENERAL BAND vs FAULT ORDER (training, higher = better) ===\n")
    for label, feats_ in [("GENERAL BAND", GEN_FEATS), ("FAULT ORDER ", FO_FEATS),
                          ("BOTH combined", GEN_FEATS + FO_FEATS)]:
        his = build_hi(data, feats_)
        monos = [_mono(his[tr]) for tr in data]
        trends = [_trend(his[tr]) for tr in data]
        print(f"[{label}] mono: " + " ".join(f"T{tr}={m:.2f}" for tr, m in zip(data, monos)) +
              f" | mean={np.mean(monos):.3f}")
        print(f"[{label}] trend:" + " ".join(f"T{tr}={t:.2f}" for tr, t in zip(data, trends)) +
              f" | mean={np.mean(trends):.3f}\n")

    # late/early discrimination per family (mean over channels & bearings)
    print("=== late/early HI-feature contrast (degradation visibility) ===")
    for label, feats_ in [("GENERAL BAND", GEN_FEATS), ("FAULT ORDER ", FO_FEATS)]:
        ratios = []
        for tr, df in data.items():
            nh = max(3, int(len(df) * 0.15))
            for c in range(4):
                for f in feats_:
                    col = f"Ch{c}_{f}"
                    if col in df.columns:
                        early = df[col].iloc[:nh].mean()
                        late = df[col].iloc[-nh:].mean()
                        ratios.append(late - early)   # log-scale: difference = log ratio
        print(f"[{label}] mean log10(late/early) = {np.mean(ratios):+.3f}  "
              f"(x{10**np.mean(ratios):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bearing", type=int)
    ap.add_argument("--test", type=str, help="validation bearing name e.g. Test1")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(); return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.test:
        (OUT / "test").mkdir(parents=True, exist_ok=True)
        df = extract(None, test_name=args.test)
        df.to_csv(OUT / "test" / f"{args.test}.csv", index=False)
        print(f"{args.test}: {len(df)} files, {time.time()-t0:.0f}s", flush=True)
    else:
        df = extract(args.bearing)
        df.to_csv(OUT / f"Train{args.bearing}.csv", index=False)
        print(f"Train{args.bearing}: {len(df)} files, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
