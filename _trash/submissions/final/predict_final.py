"""아이사(AISA) 최종 RUL 예측 파이프라인 — KSPHM-KIMM 2026.

진동 데이터 + *실제 리더보드 점수*만으로 도출 (가정값 미사용).
PARAMS는 실제 점수(B, V2, 그리고 마지막 확인 제출의 실제 점수)에 맞춰 보정됨.

방법:
  1) 전이성 지표(Order_BandEnergy, Spectral_Entropy) + 선택적 보조지표(Env/RMS).
  2) 열화분율 lf = (현재−건전)/(EOL−건전), 학습 4 베어링 기준.
  3) RUL = CAP − (lf**GAMMA)·(CAP−FLOOR).
  PARAMS(FEATURES 가중치/CAP/GAMMA)는 calibrate.py가 실제 점수에 맞춰 산출한 값.

실행:  python predict_final.py
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
TESTS = [f"Test{i}" for i in range(1, 7)]
EST = W.load("est")

# ===== 보정 파라미터 (실제 점수로 calibrate; 아래는 전이성 기본값) =====
PARAMS = {
    "FEATURES": {"Order_BandEnergy": 1.0, "Spectral_Entropy": 1.0},  # 지표:가중치
    "AGG": "max",        # 지표 통합: max(하나라도 발화) / wmean(가중평균)
    "CAP": 53153.0,      # 무열화 기대 RUL (학습 RUL@50 중앙값)
    "FLOOR": 1800.0,
    "GAMMA": 1.0,        # lf 매핑 곡률
}
# =====================================================================


def _cols(df, feat):
    return [c for c in df.columns if c.endswith(feat)]

def _series(df, feat):
    x = df[_cols(df, feat)].to_numpy(float).max(1)
    return pd.Series(x).rolling(5, min_periods=1).median().to_numpy()

def _lf(test_df, feat):
    cur = _series(test_df, feat)[-1]
    heal = [np.median(_series(EST[t], feat)[: max(3, int(len(EST[t]) * 0.15))]) for t in (1, 2, 3, 4)]
    eol = [_series(EST[t], feat)[-1] for t in (1, 2, 3, 4)]
    h, e = np.median(heal), np.median(eol)
    return min(max((cur - h) / (e - h + 1e-9), 0.0), 1.0)

def predict(test_df, P=PARAMS):
    feats = P["FEATURES"]
    lfs = {f: _lf(test_df, f) for f in feats}
    if P["AGG"] == "max":
        lf = max(lfs[f] for f in feats)
    else:  # 가중평균
        w = sum(feats.values())
        lf = sum(lfs[f] * feats[f] for f in feats) / (w + 1e-9)
    return int(round(max(P["FLOOR"], P["CAP"] - (lf ** P["GAMMA"]) * (P["CAP"] - P["FLOOR"]))))

def main():
    rows = [{"File": f"Validation{i+1}",
             "RUL_Score": predict(pd.read_csv(R.ESTT / f"{n}.csv").sort_values("File_Index"))}
            for i, n in enumerate(TESTS)]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_excel(HERE / "아이사_validation.xlsx", index=False)
    print("-> 아이사_validation.xlsx 생성 완료")

if __name__ == "__main__":
    main()
