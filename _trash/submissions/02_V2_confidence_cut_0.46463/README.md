# 제출 V2 (둘째날) — 실제 점수 0.46463255310022616

벡터: `[53153, 51138, 13079, 35633, 14732, 14313]`  (Validation1..6)

## 성격: 모델 + 비율 보정 + Validation2 손교체 (로직 전체 재현됨)
`predict_v2.py` 는 이제 최종 숫자를 하드코딩하지 않고 **실제 생성 로직을 그대로 계산**한다:

1. **base** = robust 앙상블 모델 RUL — 베어링별 4개 지표(Spectral/Order/Env/RMS)의
   열화분율 lf 평균 → `RUL = CAP - mean_lf*(CAP-FLOOR)`. (scripts/predict_robust.py 로직 재사용)
   → `[53153, 24784, 22168, 41921, 28886, 27005]` (데이터에서 계산, 검증됨)
2. **std** = 그 4개 지표 lf의 표준편차 = 지표 불일치 = 불확실도.
3. **conf** = `round(1 - std, 2)` — 신뢰확률(불확실할수록 작음).
4. **v** = `round(base * conf)` — 잘못 예측할 가능성만큼 비율로 깎는 보정.
5. **v[1] = 51138** — Validation2만 손으로 B값으로 교체(첫날 0.49에서 Test2가 길었다는 근거).

이게 어제 실제로 한 그대로다: *모델 → 일정 비율(=1-불확실도) 보정 → Validation2만 교체.*

## 단독 실행 (이 폴더에서)
```
cd submissions/02_V2_confidence_cut_0.46463
python predict_v2.py
```
- 결과는 **이 폴더 안**의 `아이사_validation.xlsx` (라이브 제출파일 미수정).
- base/std를 data_challenge 데이터에서 실제 계산하므로 `scripts/predict_robust.py`와 특징 캐시(`outputs/ot_features/{est,test}`)가 있어야 함.

## 메모 (정직하게)
- 원본 인라인 명령(`_original_command.txt`)은 base/std를 **하드코딩**해 적었고, 거기서
  Test2 std를 0.39로 표시했다. 데이터 재계산 시 Test2 std = 0.40 으로 나오지만,
  **Test2는 5)에서 51138로 덮어쓰므로 최종 결과는 동일**. 나머지 베어링 std(0.41/0.15/0.49/0.47)는 정확히 일치.
- Validation2 교체는 모델이 아니라 **사람 판단**이다. VAL2_OVERRIDE 상수로 명시해 뒀다.

## 검증됨 (2026-06-04)
폴더 단독 실행 시 정확히 `[53153, 51138, 13079, 35633, 14732, 14313]` 재현 (MATCH: True).
