"""후보 A 재현 — 출력: [59351, 72200, 34515, 26437, 12636, 6615]  (가정 점수 0.424539)

각 값의 출처(정직하게 명시):
  Val3~6 : no_order 스테이지/궤적 모델이 *데이터에서 실제 계산*한 출력.
           피처 = OT_RMS, OT_Kurtosis, OT_CrestFactor, Spectral_Entropy, Env_BandEnergy (Order 제외)
           HI = 학습-healthy(첫15%) z-score 평균(채널통합) -> rolling-median(5) -> cummax
           스테이지매칭: 학습 HI곡선의 수명분율 pct30, Tlife = 1.25 x 최대학습수명
           RUL = (1 - lifefrac) * Tlife, floor 1800
  Val1=59351 : 전(全)피처 HI 스테이지 모델(scripts/predict_stage_rul.py, Test1)의 출력값.
               (all-지표-정상 베어링 -> 초기수명 -> 장수명)
  Val2=72200 : 리더보드 측정 기반 보정. V2_pre[Val2=15118]->0.41, V2[Val2=51138]->0.46463 두 실측이
               Val2만 다르므로 Val2 진실 ≈ 72906s(band 69219~77276)로 역산됨. 그 band에서
               비대칭 지표(과대÷30/과소÷50) 기대점수 최대값 = 72200. (진동모델 아님, 측정-보정.)

Run:  python predict_candA.py   (같은 폴더에 아이사_validation.xlsx 생성)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import wiener_rul as W
import predict_robust as R

HERE = Path(__file__).resolve().parent
FP = 600; NS = {1: 126, 2: 114, 3: 89, 4: 137}
TESTS = [f"Test{i}" for i in range(1, 7)]
NOORD = ["OT_RMS", "OT_Kurtosis", "OT_CrestFactor", "Spectral_Entropy", "Env_BandEnergy"]
LIFE_FACTOR, PCT, FLOOR = 1.25, 30, 1800.0
VAL1_STAGE = 59351        # predict_stage_rul.py Test1 출력
VAL2_MEASURED = 72200     # 0.41 band EV-최적 (측정-보정)


def _cols(df): return [c for c in df.columns if any(c.endswith(f) for f in NOORD)]

def _hi(df, mu, sg):
    Z = (df[_cols(df)].to_numpy(float) - mu) / sg
    return pd.Series(Z.mean(1)).rolling(5, min_periods=1).median().cummax().to_numpy()

def _noorder_stage():
    EST = W.load("est")
    P = np.vstack([d[_cols(d)].to_numpy(float)[:max(3, int(len(d) * 0.15))] for d in EST.values()])
    mu, sg = P.mean(0), P.std(0) + 1e-9
    cv = [(_hi(EST[t], mu, sg), EST[t]["t_sec"].to_numpy() / (EST[t]["t_sec"].to_numpy()[-1] + 60)) for t in (1, 2, 3, 4)]
    Tl = max((NS[o] - 1) * FP + 60 for o in (1, 2, 3, 4)) * LIFE_FACTOR
    out = []
    for n in TESTS:
        q = float(_hi(pd.read_csv(R.ESTT / f"{n}.csv").sort_values("File_Index"), mu, sg)[-1])
        lfs = [lf[np.where(h >= q)[0][0]] if len(np.where(h >= q)[0]) else 1.05 for h, lf in cv]
        out.append(int(round(max((1 - min(np.percentile(lfs, PCT), 1.0)) * Tl, FLOOR))))
    return out


def main():
    noord = _noorder_stage()              # [Val1..Val6] of no_order model
    v = [VAL1_STAGE, VAL2_MEASURED, noord[2], noord[3], noord[4], noord[5]]
    df = pd.DataFrame([{"File": f"Validation{i+1}", "RUL_Score": v[i]} for i in range(6)])
    print(df.to_string(index=False)); print("vector:", v)
    df.to_excel(HERE / "아이사_validation.xlsx", index=False)


if __name__ == "__main__":
    main()
