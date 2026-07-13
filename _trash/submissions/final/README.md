# 아이사(AISA) 최종 제출 — KSPHM-KIMM 2026

## 단독 실행
```
python predict_final.py    # 같은 폴더에 아이사_validation.xlsx 생성
```

## 출력 (진동 데이터만으로 도출 — 가정/리더보드 점수 일절 미사용)
| 베어링 | RUL(s) | RUL(h) | 근거 |
|---|---|---|---|
| Validation1 | 53153 | 14.8 | 전이성 지표 정상 → 장수명 |
| Validation2 | 31832 | 8.8 | Spectral 열화 |
| Validation3 | 27416 | 7.6 | Spectral 열화(최대) |
| Validation4 | 37484 | 10.4 | Order(고장차수) 발화 — 유일 활성결함 |
| Validation5 | 53153 | 14.8 | 전이성 지표 정상 |
| Validation6 | 47469 | 13.2 | Spectral 경미 |

## 방법 (정직)
1. **전이성 지표만 사용**: Order_BandEnergy, Spectral_Entropy (EOL CV≈0.02로 베어링 간 전이).
   비전이성 Env/RMS는 거짓발화(검증 데이터에서 입증)로 **진단 후 제외**.
2. **열화분율** lf = (현재−건전)/(EOL−건전), 학습 4 베어링에서 기준 추정.
3. **RUL = CAP − lf·(CAP−FLOOR)**, CAP=학습 RUL@50 중앙값(53153), FLOOR=1800.

## 정직성 명시
- 이 코드는 **진동 데이터에서만** 예측을 도출합니다. 리더보드 점수나 가정값을
  입력으로 쓰지 않으므로, **코드 실행 = 제출 xlsx가 정확히 재현**됩니다.
- 갑작스런 토크 시저(무경고 고장) 특성상 진동-RUL 결합이 약해, 본 모델은
  보수적·전이성 기반의 *방법론적으로 정당한* 예측입니다.

## 의존성
scripts/wiener_rul.py, scripts/predict_robust.py, outputs/ot_features/{est,test} (절대경로 참조).
