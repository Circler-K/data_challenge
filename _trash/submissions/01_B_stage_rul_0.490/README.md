# 제출 B (첫날) — 실제 점수 0.49

벡터: `[59351, 51138, 19936, 12168, 1800, 1800]`  (Validation1..6)

## 성격: 진짜 모델 (하드코딩 아님)
`predict_stage_rul.py` 는 HI 데이터로부터 59351, 51138… 을 **실제로 계산**한다.
HI 단계(stage) 기반: 각 검증 베어링 HI 마지막값을 학습 베어링들이 그 HI에 도달하는
수명분율(50퍼센타일)에 매칭 → RUL = (1 − 수명분율) × 중앙수명(19.9h), 하한 1800초.

## 단독 실행 (이 폴더에서)
```
cd submissions/01_B_stage_rul_0.490
python predict_stage_rul.py
```
- 결과는 **이 폴더 안**의 `아이사_validation.xlsx`, `test_rul_stage.csv` 에 쓴다(라이브 제출파일 안 건드림).
- cwd 무관(절대경로 사용). 원본 명령/원본 출력경로는 `_original_command.txt` 참고.
- 모델/계산 로직은 원본과 한 줄도 다르지 않음. 출력 경로만 이 폴더로 바꿈.

## 의존성 (data_challenge 리포가 그대로 있어야 함)
- `scripts/wiener_rul.py`, `scripts/predict_test_wiener.py`, `scripts/rpm_estimator.py`, `scripts/ot_rpm_impact.py`
- `outputs/ot_features/{est, test}/` (특징 캐시), `nptdms` 패키지
→ 데이터/모델을 data_challenge에서 절대경로로 읽으므로, 리포가 살아있는 한 어디서 실행하든 동일 결과.

## 검증됨 (2026-06-04)
폴더 단독 실행 시 정확히 `[59351, 51138, 19936, 12168, 1800, 1800]` 재현.
