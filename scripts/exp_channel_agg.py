"""채널 집계 방식 실험 — 전이성 풀에서 HI 집계를 학습 file-50 LOO로 선택.

누수 없음: 후보는 모두 학습 EOL_CV<0.05 전이성 지표로만 구성, 선택은 file-50 LOO로만.
각 후보의 LOO + 결과 Test 벡터를 출력 → 최고 LOO 후보가 V3/V4를 잡는지 확인.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EST = ROOT / "outputs" / "ot_features" / "est"
TEST = ROOT / "outputs" / "ot_features" / "test"
TESTS = [f"Test{i}" for i in range(1, 7)]
STOP = {1: 75251, 2: 67979, 3: 53225, 4: 82613}
FP = 600
CAP, FLOOR = 53153.0, 3600.0
CALIB_CUTS = (45, 50, 55)
GAMMAS = (1.0, 1.5, 2.0, 3.0, 4.0)

def load(p): return pd.read_csv(p).sort_values("File_Index").reset_index(drop=True)
def smooth(v): return pd.Series(v).rolling(5, min_periods=1).median().to_numpy()
def a_rul(act, pred):
    er = 100.0 * (act - pred) / act; ln = np.log(0.5)
    return float(np.exp(-ln * er / 30.0) if er <= 0 else np.exp(ln * er / 50.0))
def actrul(t, c): return STOP[t] - ((c - 1) * FP + 60)

TR = {t: load(EST / f"Train{t}.csv") for t in (1, 2, 3, 4)}
TE = {n: load(TEST / f"{n}.csv") for n in TESTS}

def col_series(df, col): return smooth(df[col].to_numpy(float))
def maxcol_series(df, feat):
    return smooth(df[[c for c in df.columns if c.endswith(feat)]].to_numpy(float).max(1))
def healthy(s): return float(np.median(s[:max(3, int(len(s) * 0.15))]))

# 후보 = 지표 채널 리스트 (각 원소: ('max', feat) 또는 ('ch', colname))
CANDS = {
    "M0 현행 mean[Spec_max,Ord_max]": [("max", "Spectral_Entropy"), ("max", "Order_BandEnergy")],
    "M1 +Ch2/Ch3_Ord mean": [("max", "Spectral_Entropy"), ("max", "Order_BandEnergy"), ("ch", "Ch2_Order_BandEnergy"), ("ch", "Ch3_Order_BandEnergy")],
    "M2 +Ch2/Ch3_Ord median": [("max", "Spectral_Entropy"), ("max", "Order_BandEnergy"), ("ch", "Ch2_Order_BandEnergy"), ("ch", "Ch3_Order_BandEnergy")],
    "M3 mean[Spec_max,Ch2_Ord,Ch3_Ord]": [("max", "Spectral_Entropy"), ("ch", "Ch2_Order_BandEnergy"), ("ch", "Ch3_Order_BandEnergy")],
    "M4 median[Spec_max,Ord_max,Ch2,Ch3,Ch0_Spec]": [("max", "Spectral_Entropy"), ("max", "Order_BandEnergy"), ("ch", "Ch2_Order_BandEnergy"), ("ch", "Ch3_Order_BandEnergy"), ("ch", "Ch0_Spectral_Entropy")],
    "M5 perchan-max[Order]": [("pcmax", "Order_BandEnergy")],
    "M6 mean[Order_pcmax, Spec_pcmax]": [("pcmax", "Order_BandEnergy"), ("pcmax", "Spectral_Entropy")],
    "M7 mean[Order_pcmax, Spec_max]": [("pcmax", "Order_BandEnergy"), ("max", "Spectral_Entropy")],
}
AGG = {"M2 +Ch2/Ch3_Ord median": np.median, "M4 median[Spec_max,Ord_max,Ch2,Ch3,Ch0_Spec]": np.median}

CHS = ["Ch0", "Ch1", "Ch2", "Ch3"]
def comp(df, item):
    return maxcol_series(df, item[1]) if item[0] == "max" else col_series(df, item[1])

def perchan_hi(df, feat, baseline_dfs):
    """채널별로 먼저 정규화한 HI를 만든 뒤 채널 max (고장-채널 무관)."""
    his = []
    for ch in CHS:
        col = f"{ch}_{feat}"
        h = np.median([healthy(col_series(b, col)) for b in baseline_dfs])
        e = np.median([col_series(b, col)[-1] for b in baseline_dfs])
        his.append(np.clip((col_series(df, col) - h) / (e - h + 1e-9), 0, 1))
    return np.max(his, 0)

def HI(df, items, baseline_dfs, agg):
    parts = []
    for it in items:
        if it[0] == "pcmax":
            parts.append(perchan_hi(df, it[1], baseline_dfs))
            continue
        h = np.median([healthy(comp(b, it)) for b in baseline_dfs])
        e = np.median([comp(b, it)[-1] for b in baseline_dfs])
        parts.append(np.clip((comp(df, it) - h) / (e - h + 1e-9), 0, 1))
    return smooth(agg(parts, 0))

def degfrac(hi, g): return max(FLOOR, CAP - (hi ** g) * (CAP - FLOOR)) * 0.9

def loo_score(items, agg, g):
    scs = []
    for h in (1, 2, 3, 4):
        others = [TR[o] for o in (1, 2, 3, 4) if o != h]
        for c in CALIB_CUTS:
            hi = HI(TR[h].iloc[:c], items, others, agg)[-1]
            scs.append(a_rul(actrul(h, c), degfrac(hi, g)))
    return float(np.mean(scs))

print(f"{'후보':46s} {'LOO':>6s} {'g':>4s} |  Test 벡터(h)  + HI")
print("-" * 100)
results = []
for name, items in CANDS.items():
    agg = AGG.get(name, np.mean)
    best = max(((loo_score(items, agg, g), g) for g in GAMMAS), key=lambda x: x[0])
    loo, g = best
    allb = [TR[o] for o in (1, 2, 3, 4)]
    his = [HI(TE[n], items, allb, agg)[-1] for n in TESTS]
    vec = [int(round(degfrac(hi, g))) for hi in his]
    results.append((loo, name, g, vec, his))

results.sort(reverse=True)
for loo, name, g, vec, his in results:
    hh = " ".join(f"{h:.2f}" for h in his)
    vh = " ".join(f"{v/3600:4.1f}" for v in vec)
    print(f"{name:46s} {loo:6.3f} {g:4.1f} | {vh}")
    print(f"{'':46s} {'HI:':>11s} {hh}")
    print(f"{'':46s} sec= {vec}")
print("\n[판정] 최고 LOO 후보가 현행(M0)보다 높고 V3·V4 HI를 키우면 → 정당한 개선.")
