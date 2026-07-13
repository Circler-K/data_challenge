"""V3(·V4) 열화 탐지 가능성 진단 — 누수 없는 지표 스크리닝.

선택 기준은 **학습 4궤적의 전이성(EOL_CV)** 만. 그 뒤 각 지표가 6개 Test의
마지막 파일을 얼마나 열화로 보는지(HI) 출력 → 전이성 지표가 V3/V4를 독립적으로
단수명으로 지목하는지 확인. (앵커/정답 미사용)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EST = ROOT / "outputs" / "ot_features" / "est"
TEST = ROOT / "outputs" / "ot_features" / "test"
FEATS = ["OT_RMS", "OT_Kurtosis", "OT_CrestFactor", "Order_BandEnergy", "Spectral_Entropy", "Env_BandEnergy"]
CHS = ["Ch0", "Ch1", "Ch2", "Ch3"]
TRAINS = [1, 2, 3, 4]
TESTS = [f"Test{i}" for i in range(1, 7)]

def load(p): return pd.read_csv(p).sort_values("File_Index").reset_index(drop=True)
def smooth(v): return pd.Series(v).rolling(5, min_periods=1).median().to_numpy()
TR = {t: load(EST / f"Train{t}.csv") for t in TRAINS}
TE = {n: load(TEST / f"{n}.csv") for n in TESTS}

def series(df, cols, agg):
    M = df[cols].to_numpy(float)
    v = M.max(1) if agg == "max" else M.mean(1) if agg == "mean" else M[:, 0]
    return smooth(v)

def healthy(s): return np.median(s[:max(3, int(len(s) * 0.15))])

def screen(cols_for, label, agg):
    """cols_for(train_or_test_df)->columns. 전이성(EOL_CV) + Test HI."""
    # 학습 EOL 값 (마지막 파일, smoothed) across trains → 전이성
    eols, healths = [], []
    for t in TRAINS:
        s = series(TR[t], cols_for(TR[t]), agg)
        eols.append(s[-1]); healths.append(healthy(s))
    eols = np.array(eols); healths = np.array(healths)
    eol_cv = float(np.std(eols) / (abs(np.mean(eols)) + 1e-9))
    # 모델식 HI: h,e = 전체 학습 중앙값 (LOO 아님, 스크리닝용 근사)
    h = float(np.median(healths)); e = float(np.median(eols))
    his = {}
    for n in TESTS:
        s = series(TE[n], cols_for(TE[n]), agg)
        his[n] = float(np.clip((s[-1] - h) / (e - h + 1e-9), 0, 1))
    return eol_cv, his

print(f"{'지표':28s} {'EOL_CV':>7s} | " + " ".join(f"{n[-1]:>5s}" for n in TESTS) + "   (HI: 1=열화)")
print("-" * 80)
rows = []
# (a) 채널 max 집계 (모델 방식)
for f in FEATS:
    cv, his = screen(lambda df, f=f: [f"{c}_{f}" for c in CHS], f"{f}(max)", "max")
    rows.append((f"{f}(ch-max)", cv, his))
# (b) 채널별 단독 — 특정 채널이 V3를 잡는지
for f in FEATS:
    for ch in CHS:
        cv, his = screen(lambda df, ch=ch, f=f: [f"{ch}_{f}"], f"{ch}_{f}", "single")
        rows.append((f"{ch}_{f}", cv, his))

# 전이성 좋은 순으로 정렬 출력
rows.sort(key=lambda r: r[1])
for name, cv, his in rows:
    flag = ""
    if cv < 0.10:  # 전이성 양호 후보
        if his["Test3"] > 0.5: flag += " <V3잡음"
        if his["Test4"] > 0.5: flag += " <V4잡음"
    print(f"{name:28s} {cv:7.3f} | " + " ".join(f"{his[n]:5.2f}" for n in TESTS) + flag)

print("\n[해석] EOL_CV<0.10 = 학습 전이성 양호. 그 중 Test3 HI>0.5 인 지표가 있으면")
print("       V3 열화를 누수 없이 잡을 수 있다는 뜻. 없으면 V3 단수명은 전이성 신호로 미검출.")
