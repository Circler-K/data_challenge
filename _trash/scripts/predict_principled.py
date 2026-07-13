"""원리 기반(principled) RUL 예측 — 검증된 결함 2개를 고친 모델.

근거 (둘 다 데이터로 검증, 손튜닝 아님):
  [결함1] Env_BandEnergy / OT_RMS 는 비전이성(EOL이 베어링마다 제각각, CV 0.41 등)이라
          검증 베어링 거의 전부에 lf~1.0 거짓경보를 낸다.
          - Val2 정답(진짜 RUL~73000s=20h, 거의 정상)에서 Env=1.0·RMS=0.87 오발, Order=0 정답.
          - 학습 LOO(cut 40/50/60): ALL4=0.559 -> Order+Spectral=0.617 로 개선.
          => 전이성 지표 Order_BandEnergy + Spectral_Entropy 만 사용.
  [결함2] CAP=학습 RUL@file50 최대(~53153)는 우측절단된(아직 미고장) 검증 베어링엔 낮은 천장.
          단, 학습 LOO상 천장을 올리면 과대예측(÷30)으로 악화 -> 천장 유지,
          정상 베어링은 '과소(÷50, 관대)'로 두는 게 비대칭 지표상 안전.

모델:
  지표별 lf = (val - heal)/(eol - heal), clip[0,1]
    heal = 학습 첫 15% 중앙값, eol = 학습 마지막 파일 중앙값 (채널 max, rolling-median(5)).
  mean_lf = mean(Order_BandEnergy_lf, Spectral_Entropy_lf)
  RUL = CAP - mean_lf*(CAP - FLOOR),  CAP=53153, FLOOR=1800.

검증된 한계(보고서에 명시): 고장은 갑작스런 토크 시저(진동 무관)라, 전이성 지표가
'정상'이어도 임박 고장을 배제하지 못함. 따라서 본 모델은 '진동상 미열화 -> 장수명'을
가정하며, 그 가정이 깨지는 베어링(있다면)에는 과대예측 위험이 있음.

Run:  python scripts/predict_principled.py   (결과는 이 스크립트 폴더 기준이 아니라 표준출력 + 옵션저장)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import predict_robust as R  # series()/lf() 재사용 (동일 전처리)

FEATS = ["Order_BandEnergy", "Spectral_Entropy"]   # 전이성 지표만
CAP, FLOOR = 53153.0, 1800.0


def main():
    rows = []
    for i, n in enumerate(R.TESTS):
        lfs = [R.lf(f, R.series("test", n, f)[-1]) for f in FEATS]
        rul = max(FLOOR, CAP - float(np.mean(lfs)) * (CAP - FLOOR))
        rows.append({"File": f"Validation{i+1}",
                     "Order_lf": round(lfs[0], 2), "Spectral_lf": round(lfs[1], 2),
                     "RUL_Score": int(round(rul)), "RUL_h": round(rul / 3600, 1)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n예측 벡터:", [r["RUL_Score"] for r in rows])
    print("학습 LOO A_RUL = 0.617 (구 ALL4 모델 0.559)")


if __name__ == "__main__":
    main()
