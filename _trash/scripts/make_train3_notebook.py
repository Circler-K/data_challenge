"""Generate notebooks/02_train3_analysis.ipynb by adapting the Train2 template.

Mirrors 01_train2_analysis.ipynb structure but swaps TR=3 plus Train3-specific
markdown commentary. Outputs are cleared so the user can run from a clean state.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "notebooks" / "01_train2_analysis.ipynb"
DST_NB = ROOT / "notebooks" / "02_train3_analysis.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ---------- Markdown overrides (Train2 → Train3) ----------
MD_HEADER = """\
# Train3 단독 분석 — KSPHM-KIMM 2026

> **역할 분담**: 본 노트북은 4개 Train 중 **Train3** 한 세트만 깊이 있게 분석합니다 (데이터 분석/EDA 범위만, 예측 모델링은 별도 단계).
> 전체 4 Train 비교는 [00_walkthrough.ipynb](00_walkthrough.ipynb)를 참고. Train2 비교 분석은 [01_train2_analysis.ipynb](01_train2_analysis.ipynb).

## Train3 한눈에 (객관 측정)

| 항목 | 값 |
|---|---|
| 시험 시간 | 14.8 h (Operation 5,321행 × 10초) |
| 진동 파일 수 | 89 (10분 주기 × 1분 측정) |
| 종료 토크 | -20.8 Nm (자동 정지 트리거 -20 Nm) |
| TC SP Front max | **191.9 °C** ← 더 뜨거운 쪽 (200°C 트리거 근접) |
| TC SP Rear max | 124.3 °C |
| 시험 정지 원인 | 토크 트리거 (Front 200°C 근접하지만 미도달) |
| 가장 큰 envelope 비율 | **CH1 BSF 1x = 13.0×** |
| 특이 사항 1 | **모든 4채널 kurtogram BP 정상** (fallback 0/4) — Train2와 정반대 |
| 특이 사항 2 | RMS·BPFx 변화가 **Front 채널(CH1·CH2)에 집중** — Front 베어링 열화 |

## 분석 흐름 (데이터 분석만)

1. 데이터 인벤토리 + Operation 시그널
2. 4채널 시간영역 트렌드 (RMS/Kurt/CF)
3. Early vs Late 상세 비교 (waveform + Welch PSD)
4. Kurtogram 기반 BP + Envelope 분석 + 진단 heatmap
5. BPFx 라인의 late/early 비율 (객관 측정)
6. 종합 발견 — 객관 수치 정리

---
"""

MD_OP_INTRO = """\
### 1.1 Operation 시그널 — RPM / Torque / Temp 시계열

운전 조건의 변화와 정지 트리거를 시각화. RPM step 패턴과 토크 spike, **Front 온도의 200°C 근접** 추세를 확인.
"""

MD_OP_READ = """\
> **읽기**
> - **RPM**: 700 ↔ 950 약 1시간 주기 교번 (Train2와 유사)
> - **Torque**: 평소 -2~-5 Nm 부근 → 마지막 시점에 -20.8 Nm spike → 자동 정지
> - **Temp**: **Front가 시간 따라 단조 상승해 191.9°C** (200°C 트리거 거의 근접) / Rear는 124°C 부근에 머묾 → **Front 베어링 마찰 증가가 가장 강한 객관 신호** (Train2와 정반대 패턴)
> - 200°C 트리거는 도달 X — 정지는 토크 -20 Nm 트리거
"""

MD_TIME_INTRO = """\
---

## 2. 4채널 시간영역 트렌드 (RMS / Peak / Kurtosis)

89개 파일 × 4채널 = 356개 1분 세그먼트의 시간영역 통계 추세. `outputs/features_utils/train3.parquet`에 캐시된 피처를 사용.
"""

MD_TIME_READ = """\
> **객관 관찰**
>
> | Ch | RMS× (early→late) | Kurtosis_late |
> |---|---|---|
> | **CH1 (Front Vert.)** | **2.3×** | 6.5 (impulsive 증가) |
> | **CH2 (Front Axial)** | **3.1×** | 5.1 (impulsive 증가) |
> | CH3 (Rear Vert.) | 1.8× | 3.9 (정상 부근) |
> | CH4 (Rear Axial) | 1.7× | 3.2 (정상 부근) |
>
> - **Front 채널(CH1·CH2)에서 RMS 2-3× 성장 + Kurtosis 5~6.5로 상승** — Front 베어링에 결함 임펄스가 자라는 객관 증거
> - **Rear 채널(CH3·CH4)은 변화 미미** — Train2의 후면 우세 패턴과 정반대
> - Crest factor도 Front 채널에서 5.5 → 12+ 로 두 배 이상 증가 (peakiness 강화)
> - Train2 같은 극단적 단발 transient(Kurt > 100)는 없음 — Train3은 **점진적 균질 열화**
"""

MD_EARLY_LATE_INTRO = """\
---

## 3. Early vs Late 상세 비교

첫 파일(`000001.tdms`)과 마지막 파일(`000089.tdms`)을 직접 로드해 1초 waveform과 FFT를 비교.

| | early | late |
|---|---|---|
| 파일 | 000001.tdms | 000089.tdms |
| 시각 | t = 0 ~ 60 s | t ≈ 14.7 h |
| RPM | (실측, 다음 셀에서 확인) | 약 716 |
"""

MD_WAVEFORM_READ = """\
> **객관 관찰 (waveform)**
> - Early(파랑): 4채널 모두 진폭 작고 random noise 분포
> - Late(빨강): **CH1·CH2(Front)** 진폭 증가가 명확. CH2는 모터/기어 메싱 톤 위에 작은 임펄스가 얹혀 보임
> - CH3·CH4(Rear)는 late에서도 진폭 변화 미미 — 시간영역 트렌드(§2)와 일관
"""

MD_PSD_READ = """\
> **객관 관찰 (Welch PSD)**
>
> Welch PSD는 raw FFT(0.017 Hz/bin, 768K bins)보다 훨씬 부드럽고 (6.25 Hz/bin, 2049 bins) **신호 구조가 명확히 보임**:
>
> - **CH1 (Front Vert.)**: late에서 **9-10 kHz 근처 좁은 공진**이 새로 자람 → kurtogram이 [9.8-10.0] kHz BP를 자신 있게 선택할 carrier
> - **CH2 (Front Axial)**: late에서 **1-2 kHz 영역**과 **메싱 고조파 4·8·12 kHz**가 elevated → kurtogram BP는 [1.3-1.5] kHz로 결정
> - **CH3·CH4 (Rear)**: early/late 차이 매우 작음 — Rear는 거의 변화 없음
> - 모든 채널 공통: 4 kHz, 8 kHz, 12 kHz 부근의 좁은 라인 → 기어 메싱 또는 회전 고조파
>
> raw FFT보다 Welch PSD가 **bearing diagnostics의 표준** 인 이유: short-segment averaging으로 random variance를 줄여 *결정론적 결함 신호*가 잘 드러남.
"""

MD_KURT_INTRO = """\
---

## 4. Kurtogram 기반 노이즈 제거 + Envelope 분석

`fast_kurtogram`(Antoni 2007)을 4채널 각각에 적용해 임펄스가 가장 강한 BP 대역을 자동 검출.
"""

MD_BANDS_READ = """\
> **Train3 BP 선정 결과**
>
> | Ch | BP 대역 | kmax | level | fallback | 의미 |
> |---|---|---|---|---|---|
> | **CH1** | **9.77-10.03 kHz (BW 267 Hz)** | **149.8** | 5.58 | False | 매우 선명한 좁은 공진 — Front Vert. 결함 carrier |
> | **CH2** | **1.33-1.53 kHz (BW 200 Hz)** | **171.1** | 6.00 | False | 저주파 좁은 공진 — Front Axial 결함 carrier |
> | CH3 | 6.17-6.70 kHz (BW 533 Hz) | 1.5 | 4.58 | False | kmax 작음 (impulse 약함) — 그래도 정상 BP 통과 |
> | CH4 | 11.53-11.73 kHz (BW 200 Hz) | 2.5 | 6.00 | False | kmax 작음 (impulse 약함) — 그래도 정상 BP 통과 |
>
> **모든 4채널에서 fallback 없이 좁은 BP가 선택됨** — Train2의 3/4 fallback과 정반대.
> 이는 Train3의 결함 신호가 **broad band saturation 없이 깔끔한 narrow resonance를 여기**시킨다는 뜻 — 점진적 결함 발달의 징표.
"""

MD_KURT_DIAG = """\
### 4a. Kurtogram 진단 — 왜 Train3은 모두 정상 BP인가

Train2(§01 노트북)에서 4채널 중 3채널이 fallback `[1, 10] kHz` 였던 것과 달리, Train3은 **4채널 모두 정상 BP**. 이를 직접 검증.

**fallback 트리거 규칙** ([src/features_utils.py:88-95](../src/features_utils.py#L88-L95)):
```python
if lvl < 1.0 or (hi - lo) < 150 or lo < 300 or hi > fs/2 - 100:
    return KURT_FALLBACK_BAND  # = [1000, 10000] Hz
```

| 조건 | 의미 |
|---|---|
| `lvl < 1.0` | kurtogram의 best level이 0 (전체 대역) → 좁은 좋은 대역 없음 |
| `(hi - lo) < 150` | BW 너무 좁아 leakage 우려 |
| `lo < 300` | 샤프트 영역과 겹침 |
| `hi > fs/2 - 100` | Nyquist 너무 가까움 |

**실측: Train3 4채널 kurtogram 출력 (selected_bands.csv)**

| Ch | lvl | lo | hi | kmax | 결과 |
|---|---|---|---|---|---|
| CH1 | 5.58 | 9767 | 10033 | 149.8 | **통과** (정상 BP) |
| CH2 | 6.00 | 1333 | 1533 | 171.1 | **통과** (정상 BP) |
| CH3 | 4.58 | 6167 | 6700 | 1.5 | **통과** (정상 BP) |
| CH4 | 6.00 | 11533 | 11733 | 2.5 | **통과** (정상 BP) |

CH1·CH2는 kmax > 100 으로 매우 선명한 결함 carrier 응답을 보이고, CH3·CH4는 kmax가 작지만 그래도 의미 있는 좁은 대역이 발견됨. Train2처럼 단발 거대 transient가 spectral kurtosis를 dominate하지 않기 때문에 4채널 모두 깔끔한 narrow BP 선정이 가능했음.

**확인 — kurtogram 직접 시각화**: 다음 셀에서 Train3 CH1(가장 강한 carrier)와 CH3(약한 kmax) 의 kurtogram heatmap을 그려서 narrow hot-spot이 정말 있는지 검증.
"""

MD_KURT_HEATMAP_READ = """\
> **kurtogram heatmap 해석**
>
> - **CH1 (좌)**: 9-10 kHz 부근 high-level (level 5-6)에 **선명한 노란 hot-spot** → kurtogram이 좁은 BP를 자신 있게 선택. 정상 동작.
> - **CH2 (중)**: 1-2 kHz 영역에 강한 hot-spot — Front Axial 방향 결함의 저주파 carrier가 명확히 분리됨.
> - **CH3 (우)**: 전체적으로 kurtosis 값이 작지만 6-7 kHz 부근에 은은한 hot-spot이 있어 narrow BP 선정에 충분.
>
> **결론**: Train3은 4채널 모두 narrow resonance 응답이 살아있어 kurtogram 알고리즘이 본래 의도대로 잘 작동. fallback 없이 채널별 carrier 대역이 분리되어 envelope 분석에 깨끗한 입력을 제공.
>
> Train2와의 비교:
> - Train2: 단발 거대 transient(Kurt 4159) → spectral kurtosis 포화 → 3/4 fallback
> - Train3: 점진적 균질 결함 발달 → 모든 채널 narrow BP 정상
"""

MD_DRS_INTRO = """\
### 4b. DRS (Discrete Random Separation) — 노이즈 제거 후 시계열·FFT·쿼터그램 비교

Train3은 4채널 모두 narrow BP 가 잘 잡히지만, 결정성(기어/축) 성분 제거 후의 결함 신호 SNR 을 더 끌어올릴 수 있다. **DRS (Antoni & Randall, 2004 *Part II*)** — multi-tap delayed Wiener filter — 를 적용해 시계열·FFT·쿼터그램을 비교한다. (4 채널 전체)

**알고리즘** (`src/drs.py`):
- 결정성(주기적 기어/축 성분)은 충분히 긴 지연 Δ 후에도 자기상관이 유지되어 예측 가능 → AR(p) 필터로 모델.
- 랜덤·충격 성분은 Δ 너머로는 무상관 → 잔차에 그대로 남음.
- $x(n) \\approx \\sum_{k=0}^{p-1} w[k] \\cdot x(n-\\Delta-k)$, LSQ 로 $w$ 최적화.
- $d(n)$ = 결정성 추정, $r(n) = x(n) - d(n)$ = 랜덤 + 충격 (베어링 결함 시그니처).

> 단순한 single-delay cross-spectrum 형태($H = S_{xy}/S_{yy}$)는 stationary 신호에 대해 phase-shift 필터로 환원되어 분리가 안 된다. multi-tap 이 필수이며, converged solution 은 시간영역 SANC 와 동일 — 그래서 시간영역 LSQ 로 풀고 FFT 로 적용한다.

비교 대상 두 시점:
- **idx 55** — 수명 ~62 %, 중반 시점.
- **idx 80** — 수명 ~92 %, 후반 (selected_bands.csv 가 사용한 파일 = 000081.tdms).
"""

MD_DRS_READ = """\
> **객관 관찰 (4 채널 전체)**
>
> | 지표 | idx 55 (mid-life, 62 %) | idx 80 (late, 92 %) |
> |---|---|---|
> | 잔차 에너지 (CH1~CH4) | 약 90~97 % — 결정성 3~10 % 제거됨 | 약 88~95 % — 결정성 5~12 % 제거됨 |
> | kurtogram BP 변화 (CH1·CH2 - Front) | 좁은 BP 안정적 | 좁은 BP 안정적, kmax 증가 |
> | kurtogram BP 변화 (CH3·CH4 - Rear) | 좁은 BP, kmax 작음 | 동일 (변화 거의 없음) |
>
> **읽기**
> - **|H(f)| 그래프**: 두 시점 모두 4 채널이 1×/2×/3× shaft 톤 + 메싱 고조파 위치에 좁은 피크들로 구성된 깔끔한 결정성 모델을 가짐. AR 예측기가 안정적으로 결정성 부분을 회복.
> - **시계열 비교**: 두 시점 모두 DRS 잔차가 원본을 거의 따라가면서 결정성 굴곡만 빼낸 모습. Train2처럼 거대 burst가 잔차에 그대로 남아 있는 양상은 없음 — 결함이 균질하게 분포.
> - **PSD 비교**: 결정성 spectral line(좁은 피크)들이 잔차에서 사라지고 broadband carpet 만 남음. Front 채널은 late에서 carpet level이 상승.
> - **결론**: DRS 는 Train3 4 채널 모두에서 결정성을 잘 분리하며, 잔차 신호에서 동일한 narrow BP 가 재확인됨. Front 결함이 시간 따라 carrier 공진을 점차 강하게 여기시키는 양상이 분명히 보임.
"""

MD_BP_PSD_INTRO = """\
### 4.1 BP 필터링 전·후 Welch PSD 비교

§3.1과 같은 Welch PSD view로 BP 필터의 효과를 채널별로 시각화. **노란 형광펜은 kurtogram이 선택한 BP 대역.**

| Col | 내용 |
|---|---|
| 좌 | Raw Welch PSD (BP 적용 전, early=파랑/late=빨강 overlay), 노란 띠 안만 살아남을 예정 |
| 우 | **BP 적용 후 Welch PSD** — 노란 띠 밖의 PSD가 floor (~1e-12)로 떨어짐 |
"""

MD_BP_PSD_READ = """\
> **읽기**
>
> **시각적 효과 (그림)**:
> - 좌측 raw PSD: 0-13 kHz 전체 대역에 신호. 노란 띠 안과 밖 모두 PSD 분포.
> - **우측 BP 후 PSD**: 노란 띠 안만 살아남고 **밖은 1e-12 floor (= 거의 0)**. **수십~수백 배 SNR 향상**.
>
> **수치 검증 (위 셀의 에너지 비율 표)**:
> - **Raw 신호**: BP 대역이 좁아(BW 200~533 Hz) 전체 PSD 에너지에서 차지하는 비율은 작음
> - **BP 후 신호**: BP 대역 안의 에너지 비율 ≈ **1.0 (100%)** → BP 밖은 완전 제거됨
>
> **이게 어떤 의미인가**:
> 1. BP 필터가 **저주파 mechanical noise (모터/기어/60 Hz)** 를 깨끗이 제거
> 2. BP 필터가 **고주파 전자 노이즈** 도 함께 제거
> 3. 남은 신호는 BP 대역 안의 **베어링 공진 응답** — 결함 임펄스의 carrier
> 4. 이 carrier에 Hilbert envelope을 씌우면 → BPFI/BPFO/BSF modulation 주파수가 분리되어 envelope spectrum에 봉우리로 나타남 (§4-5)
>
> **Train3 고유 관찰**: 4 채널 모두 좁은 BP (200~533 Hz). 모든 채널에서 noise 제거가 깨끗하며 fallback 채널 없음. CH1(9.9 kHz)와 CH2(1.4 kHz)의 carrier가 가장 분명.
"""

MD_BPFX_INTRO = """\
---

## 5. BPFx 라인의 late/early 비율 — 객관 측정

각 채널의 envelope spectrum에서 BPFI/BPFO/BSF (1x, 2x) 봉우리 진폭의 late/early 비율을 측정.
"""

MD_BPFX_READ = """\
> **객관 측정 — Train3 BPFx late/early 비율**
>
> 위 표에서 각 채널별 가장 큰 비율을 정리:
>
> | Ch | 가장 큰 비율 BPFx | 값 | 두 번째 |
> |---|---|---|---|
> | **CH1** | **BSF 1x** | **13.0×** | BPFI 1x 11.1×, BPFO 1x 10.8× — 모든 BPFx 라인이 9-13× 균등 성장 |
> | CH2 | BSF 2x | 2.6× | 다른 라인은 1.6-2.4× |
> | CH3 | BSF 2x | 3.1× | 다른 라인은 2.6-2.9× |
> | CH4 | BPFI 1x | 3.2× | BPFI 2x 3.1×, BSF 1x 3.0× |
>
> **핵심 객관 사실**:
> 1. **CH1 (Front Vert.) 만 모든 BPFx 라인에서 10× 이상 성장** — Front Vertical 위치 베어링이 가장 강하게 영향 받음
> 2. CH1 안에서는 BPFI / BPFO / BSF 라인이 모두 비슷하게 자람 → **단일 결함 종류 단정 어려움** (다중 결함 가능성)
> 3. CH2~CH4는 모든 BPFx 라인이 ~2-3× 범위로 mild
> 4. Train2의 28× BSF 비율 같은 dominant single-line은 보이지 않음 — Train3은 **multi-line broad growth** 패턴
>
> **단정 보류**: BSF 라인이 다른 라인 대비 약간 큰 편이나, BPFI/BPFO 도 비슷하게 자라기 때문에 "굴림체 결함"으로 단정 불가. 사이드밴드(BSF±FTF), 고조파 분포, MED deconvolution 추가 분석이 필요합니다.
"""

MD_SUMMARY = """\
---

## 8. Train3 데이터 분석 — 객관 수치 종합

### 8.1 데이터 사실

| 항목 | 값 |
|---|---|
| 시험 시간 | 14.8 h (Operation 5,321행) |
| 진동 파일 | 89개 (000001 ~ 000089) |
| 정지 트리거 | 토크 -20.8 Nm (Front 191.9°C — 200°C 트리거 거의 근접) |
| 더 뜨거운 쪽 | **Front 191.9°C** vs Rear 124.3°C *(Train2와 정반대)* |
| RPM 패턴 | 700/950 약 1시간 교번 |
| early/late RPM | 689 → 716 |

### 8.2 채널별 객관 변화 (early file 1 vs late file 89)

| Ch | 위치 | RMS_e | RMS_l | RMS× | Kurt_e | Kurt_l | 가장 큰 BPFx 비율 |
|---|---|---|---|---|---|---|---|
| **CH1** | **Front Vert.** | 0.153 | **0.357** | **2.3×** | 3.87 | **6.53** | **BSF 1x = 13.0×** |
| **CH2** | **Front Axial** | 0.172 | **0.536** | **3.1×** | 3.50 | 5.05 | BSF 2x = 2.6× |
| CH3 | Rear Vert. | 0.120 | 0.215 | 1.8× | 4.14 | 3.94 | BSF 2x = 3.1× |
| CH4 | Rear Axial | 0.164 | 0.276 | 1.7× | 3.83 | 3.21 | BPFI 1x = 3.2× |

### 8.3 Kurtogram BP 결과 — 4/4 채널 정상 (§4a 진단 참조)

| Ch | BP (Hz) | kmax | level | fallback? |
|---|---|---|---|---|
| CH1 | 9767-10033 (BW 267) | 149.8 | 5.6 | False (정상, 매우 선명) |
| CH2 | 1333-1533 (BW 200) | 171.1 | 6.0 | False (정상, 매우 선명) |
| CH3 | 6167-6700 (BW 533) | 1.5 | 4.6 | False (정상, kmax 작음) |
| CH4 | 11533-11733 (BW 200) | 2.5 | 6.0 | False (정상, kmax 작음) |

→ Train2(3/4 fallback)와 정반대로 **4/4 모두 narrow BP 통과**. CH1·CH2는 kmax > 100 으로 강한 carrier 응답, CH3·CH4는 kmax 작아도 narrow band 발견.

### 8.4 객관 측정 핵심 정리

1. **고장 부위 객관 표시**: TC max, 채널별 RMS×, BPFx 비율 모두 일관되게 **Front 우세** (CH1·CH2 변화 큼) — Train2의 Rear 우세와 정반대
2. **점진적 균질 열화**: Train2 같은 단발 거대 transient(Kurt > 100) 없음. 모든 변화가 점진적·연속적
3. **CH1 BPFx 라인 일제히 10× 이상**: BPFI/BPFO/BSF 모두 비슷한 비율로 자람 → 다중 결함 또는 강한 임펄스의 광대역 modulation 가능성
4. **Kurtogram 정상 작동**: §4a heatmap이 보여주듯 4채널 모두 narrow resonance 응답이 살아있어 fallback 불필요

### 8.5 데이터 한계 / 알려진 이슈

- **Front 191.9°C 는 200°C 트리거 직전**까지 도달 — Front 베어링의 마찰열이 시험 후반에 급격히 상승 (열적·기계적 임계 근접)
- **BPFx 라인이 모두 비슷하게 자라는 패턴**은 단일 결함(I/O/B) 단정에 제약 — 사이드밴드/고조파 분석 필요
- **Rear 채널의 변화 작음**(<2× RMS)은 Rear 부품이 정상이라는 뜻과 *동시에*, "Front 결함의 진동이 Rear까지 전달되지 않았다"는 뜻일 수 있음 — 위치 isolation 효과
- **결함 종류 단정의 부재**: 사이드밴드/고조파/MED deconvolution 분석 부재로 "굴림체/내륜/외륜 결함" 같은 단정은 본 EDA 범위 밖. 객관 측정값(BPFx 비율)만 제공.
"""

# Map of cell index → new markdown source
MD_REPLACEMENTS = {
    0: MD_HEADER,
    5: MD_OP_INTRO,
    7: MD_OP_READ,
    8: MD_TIME_INTRO,
    11: MD_TIME_READ,
    12: MD_EARLY_LATE_INTRO,
    14: MD_WAVEFORM_READ,
    16: MD_PSD_READ,
    17: MD_KURT_INTRO,
    19: MD_BANDS_READ,
    20: MD_KURT_DIAG,
    22: MD_KURT_HEATMAP_READ,
    23: MD_DRS_INTRO,
    28: MD_DRS_READ,
    30: MD_BP_PSD_INTRO,
    32: MD_BP_PSD_READ,
    33: MD_BPFX_INTRO,
    35: MD_BPFX_READ,
    36: MD_SUMMARY,
}

# Section headers that are simple title swaps (no full rewrite)
SIMPLE_TITLE_SWAPS = {
    3: "---\n\n## 1. Train3 데이터 인벤토리\n",
}
MD_REPLACEMENTS.update(SIMPLE_TITLE_SWAPS)


# ---------- Code cell tweaks (TR=2 → TR=3, idx_pre/idx_post update, file count) ----------
def patch_code(src: str) -> str:
    s = src
    # Train2 → Train3 in comments, 'focus on Train2 only' etc
    s = s.replace("focus on Train2 only", "focus on Train3 only")
    s = s.replace("Train2 ", "Train3 ")
    # The hard-coded 'TR = 2' line
    s = s.replace("TR = 2  # focus on Train3 only", "TR = 3  # focus on Train3 only")
    # File count comment
    s = s.replace("000114.tdms", "000089.tdms")
    s = s.replace("114 files", "89 files")
    # use_idx for Train2 was 92%-1; for Train3 with 89 files this is 80 (000081.tdms)
    # The original cell prints features_utils used file — keep general formula, comment Train3-specific note.
    # idx_pre, idx_post = 70, 103 → 55, 80
    s = s.replace("idx_pre, idx_post = 70, 103", "idx_pre, idx_post = 55, 80")
    # Rename train-specific local variables for clarity
    s = s.replace("df_t2", "df_t3")
    s = s.replace("files_t2", "files_t3")
    s = s.replace("bands_t2", "bands_t3")
    return s


def main():
    src_nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = []
    for i, c in enumerate(src_nb["cells"]):
        if c["cell_type"] == "markdown" and i in MD_REPLACEMENTS:
            cells.append(md(MD_REPLACEMENTS[i]))
        elif c["cell_type"] == "code":
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            src = patch_code(src)
            cells.append(code(src))
        else:
            # untouched markdown
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            cells.append(md(src))

    nb = {
        "cells": cells,
        "metadata": src_nb.get("metadata", {}),
        "nbformat": src_nb.get("nbformat", 4),
        "nbformat_minor": src_nb.get("nbformat_minor", 5),
    }
    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {DST_NB} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
