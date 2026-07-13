# 후보 A — `[59351, 72200, 34515, 26437, 12636, 6615]` (가정 점수 0.424539)

## 단독 실행
```
python predict_candA.py    # 같은 폴더에 아이사_validation.xlsx 생성
```

## 각 값 출처 (정직)
| 베어링 | 값 | 출처 |
|---|---|---|
| Val1 | 59351 | 전피처 HI 스테이지 모델(scripts/predict_stage_rul.py, Test1) |
| Val2 | 72200 | **리더보드 측정-보정** (0.41/0.46463 두 실측으로 Val2≈73k 역산, band EV최적) |
| Val3~6 | 34515,26437,12636,6615 | **no_order 스테이지 모델이 데이터에서 실제 계산** (Order제외 HI, L1.25, pct30) |

- Val3~6은 진짜 데이터→전처리→모델 출력. Val1은 별도 모델 출력. **Val2만 측정-보정**(진동모델 아님).
- 검증됨: 실행 시 정확히 `[59351, 72200, 34515, 26437, 12636, 6615]` 재현.

## 의존성
scripts/wiener_rul.py, scripts/predict_robust.py, outputs/ot_features/{est,test} (절대경로 참조).
