"""Insert §4b (DRS analysis, all 4 channels) into 01_train2_analysis.ipynb
after t2-s4a-insight. Idempotent — re-run replaces any prior s4b-* cells.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "01_train2_analysis.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))

S4B_MD = """### 4b. DRS (Discrete Random Separation) — 노이즈 제거 후 시계열·FFT·쿼터그램 비교

§4a 에서 idx 103 의 CH3/CH4 가 fallback 에 빠진 원인이 "신호 자체가 충격으로 포화"임을 보였다. 보완책으로 **DRS (Antoni & Randall, 2004 *Part II*)** — multi-tap delayed Wiener filter — 를 적용한 뒤 시계열·FFT·쿼터그램을 모두 비교한다. (4 채널 전체)

**알고리즘** (`src/drs.py`):
- 결정성(주기적 기어/축 성분)은 충분히 긴 지연 Δ 후에도 자기상관이 유지되어 예측 가능 → AR(p) 필터로 모델.
- 랜덤·충격 성분은 Δ 너머로는 무상관 → 잔차에 그대로 남음.
- $x(n) \\approx \\sum_{k=0}^{p-1} w[k] \\cdot x(n-\\Delta-k)$, LSQ 로 $w$ 최적화.
- $d(n)$ = 결정성 추정, $r(n) = x(n) - d(n)$ = 랜덤 + 충격 (베어링 결함 시그니처).

> 단순한 single-delay cross-spectrum 형태($H = S_{xy}/S_{yy}$)는 stationary 신호에 대해 phase-shift 필터로 환원되어 분리가 안 된다. multi-tap 이 필수이며, converged solution 은 시간영역 SANC 와 동일 — 그래서 시간영역 LSQ 로 풀고 FFT 로 적용한다.

합성 신호(200 Hz 톤 + 백색잡음 + 충격열) 검증: |H(200 Hz)| = 0.998 (보존), |H(7 kHz)| = 0.019 (제거), 잔차 kurtosis 16.3 ≈ 진짜 noise+impulse 16.6.

비교 대상 두 시점:
- **idx 70** — 수명 ~62 %, 충격 이벤트(파일 92~103) 직전.
- **idx 103** — 수명 ~92 %, 충격 이벤트 이후 (selected_bands.csv 가 사용한 파일)."""

S4B_CODE1 = """# DRS: multi-tap delayed Wiener filter (Antoni 2004 Part II equivalent)
from src.drs import drs as drs_fn, drs_kernel_response

DRS_DELAY = 100
DRS_P = 200

idx_pre, idx_post = 70, 103
files_t2 = list_vibration_files(TR)

drs_results = {}
for idx in [idx_pre, idx_post]:
    sig4 = tdms_to_array(load_tdms_file(str(files_t2[idx])))
    chans = {}
    for i in range(4):
        x = sig4[i].astype(np.float64)
        r, d, w = drs_fn(x, fs=FS, delay=DRS_DELAY, p=DRS_P)
        f_axis, magH = drs_kernel_response(w, delay=DRS_DELAY, n_fft=8192, fs=FS)
        chans[f'CH{i+1}'] = dict(orig=x, resid=r, det=d, w=w, fH=f_axis, magH=magH)
    drs_results[idx] = chans

# |H(f)| of the Wiener kernel — peaks at deterministic (gear/shaft) freqs, ~0 at random.
# All 4 channels at both file indices.
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
colors = {'CH1': 'tab:blue', 'CH2': 'tab:orange', 'CH3': 'tab:green', 'CH4': 'tab:red'}
for ax, idx in zip(axes, [idx_pre, idx_post]):
    for ch in ['CH1', 'CH2', 'CH3', 'CH4']:
        ax.plot(drs_results[idx][ch]['fH'], drs_results[idx][ch]['magH'],
                label=ch, lw=1.0, color=colors[ch], alpha=0.85)
    ax.set_title(f'Wiener kernel |H(f)| — file idx {idx}')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|H(f)|')
    ax.set_xlim(0, FS / 2)
    ax.set_ylim(0, 1.4)
    ax.axhline(1.0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()"""

S4B_CODE2 = """# Time-series comparison: original vs DRS residual (1s window) — 4 channels × 2 files
fig, axes = plt.subplots(4, 2, figsize=(14, 10.5), sharex='col')
for c_idx, idx in enumerate([idx_pre, idx_post]):
    # Anchor the 1-second window on the channel with the largest absolute peak
    # so the visualization captures bearing impulses when present.
    peak_ch = max(['CH1', 'CH2', 'CH3', 'CH4'],
                  key=lambda c: float(np.max(np.abs(drs_results[idx][c]['orig']))))
    x_anchor = drs_results[idx][peak_ch]['orig']
    j = int(np.argmax(np.abs(x_anchor)))
    t0 = max(0, j - FS // 2)
    t1 = min(len(x_anchor), t0 + FS)
    t_axis = np.arange(t0, t1) / FS
    for ch_i in range(4):
        ax = axes[ch_i, c_idx]
        ch = f'CH{ch_i+1}'
        ax.plot(t_axis, drs_results[idx][ch]['orig'][t0:t1], lw=0.5,
                color='steelblue', alpha=0.85, label='orig')
        ax.plot(t_axis, drs_results[idx][ch]['resid'][t0:t1], lw=0.5,
                color='crimson', alpha=0.7, label='DRS residual')
        ax.set_ylabel(ch)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if ch_i == 0:
            ax.set_title(f'idx {idx} — 1s window centered on {peak_ch} peak')
        if ch_i == 3:
            ax.set_xlabel('Time [s]')
plt.suptitle('Time-series: original vs DRS residual', fontsize=12, y=1.00)
plt.tight_layout()
plt.show()"""

S4B_CODE3 = """# Welch PSD comparison: original vs DRS residual — 4 channels × 2 files
from scipy.signal import welch as _welch

def _psd(x, fs=FS, nperseg=4096):
    f, P = _welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                  scaling='density')
    return f, P

fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True, sharey='row')
for c_idx, idx in enumerate([idx_pre, idx_post]):
    for ch_i in range(4):
        ax = axes[ch_i, c_idx]
        ch = f'CH{ch_i+1}'
        f1, P1 = _psd(drs_results[idx][ch]['orig'])
        f2, P2 = _psd(drs_results[idx][ch]['resid'])
        ax.semilogy(f1, P1, lw=0.7, color='steelblue', alpha=0.85, label='orig')
        ax.semilogy(f2, P2, lw=0.7, color='crimson', alpha=0.75, label='DRS residual')
        ax.set_xlim(0, FS / 2)
        ax.set_ylabel(ch)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        if ch_i == 0:
            ax.set_title(f'Welch PSD — idx {idx}')
        if ch_i == 3:
            ax.set_xlabel('Frequency [Hz]')
plt.suptitle('Welch PSD: original vs DRS residual', fontsize=12, y=1.00)
plt.tight_layout()
plt.show()"""

S4B_CODE4 = """# Kurtogram on all 4 channels (orig + DRS residual) at both file indices
from kurtogram import fast_kurtogram
from scipy.stats import kurtosis as _kurt
import warnings as _w

def _kgram(sig, fs=FS, nlevel=6, n_use=512_000):
    s = np.asarray(sig[:min(len(sig), n_use)], dtype=np.float64).copy()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        Kwav, Lvl, freq_w, c, kmax, bw, lvl = fast_kurtogram(s, fs, nlevel=nlevel)
    row = int(np.argmax(Kwav[np.arange(Kwav.shape[0]), np.argmax(Kwav, axis=1)]))
    j = int(np.argmax(Kwav[row, :]))
    fc = float(freq_w[j])
    return dict(kmax=float(kmax), level=float(lvl), bw=float(bw), fc=fc,
                lo=fc - bw / 2, hi=fc + bw / 2, Kwav=Kwav, freq_w=freq_w)

rows = []
kgrams = {}
for idx in [idx_pre, idx_post]:
    for ch_i in range(4):
        ch = f'CH{ch_i+1}'
        x = drs_results[idx][ch]['orig']
        r = drs_results[idx][ch]['resid']
        ko = _kgram(x); kr = _kgram(r)
        kgrams[(idx, ch, 'orig')] = ko
        kgrams[(idx, ch, 'drs')] = kr
        keep = float(np.sum(r ** 2) / np.sum(x ** 2))
        rows.append(dict(idx=idx, ch=ch,
                         kurt_orig=_kurt(x, fisher=False),
                         kurt_drs=_kurt(r, fisher=False),
                         keep_pct=keep * 100,
                         kmax_o=ko['kmax'], lvl_o=ko['level'],
                         lo_o=ko['lo'], hi_o=ko['hi'],
                         kmax_r=kr['kmax'], lvl_r=kr['level'],
                         lo_r=kr['lo'], hi_r=kr['hi']))
df_drs = pd.DataFrame(rows)
display(df_drs.round(2))

# 4 rows (channels) x 4 cols (idx70 orig, idx70 DRS, idx103 orig, idx103 DRS)
fig, axes = plt.subplots(4, 4, figsize=(18, 12))
col_specs = [(idx_pre, 'orig'), (idx_pre, 'drs'),
             (idx_post, 'orig'), (idx_post, 'drs')]
for r_i in range(4):
    ch = f'CH{r_i+1}'
    for c_i, (idx, kind) in enumerate(col_specs):
        ax = axes[r_i, c_i]
        kg = kgrams[(idx, ch, kind)]
        K = kg['Kwav']
        im = ax.imshow(K, aspect='auto', origin='lower', cmap='hot',
                       extent=[0, FS / 2, -0.5, K.shape[0] - 0.5])
        ax.set_title(f'{ch}  idx{idx} {kind}\\nkmax={kg["kmax"]:.1f} '
                     f'lvl={kg["level"]:.1f}', fontsize=9)
        if c_i == 0:
            ax.set_ylabel(f'{ch}\\nLevel')
        if r_i == 3:
            ax.set_xlabel('Frequency [Hz]')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
plt.suptitle('Kurtogram: original vs DRS residual — all 4 channels', fontsize=12)
plt.tight_layout()
plt.show()"""

S4B_INSIGHT = """> **객관 관찰 (4 채널 전체)**
>
> | 지표 | idx 70 (pre-event, 62 % 수명) | idx 103 (post-event, 92 % 수명) |
> |---|---|---|
> | 잔차 에너지 (CH1~CH4) | 약 91~98 % — 결정성 2~9 % 제거됨 | CH1/CH2 ≈ 94~96 %, CH3/CH4 ≈ **100 %** (제거할 것 없음) |
> | kurtogram BP 변화 (CH1, CH2) | 동일 또는 거의 동일 | 동일 또는 거의 동일 |
> | kurtogram BP 변화 (CH3, CH4) | narrow band 약간 이동/확장 (예: CH3 [11033,11433] → [10967,11500]) | **양쪽 모두 fallback 그대로** |
>
> **읽기**
> - **|H(f)| 그래프**: idx 70 에서 모든 채널이 1×/2×/3× shaft 톤 부근 좁은 피크들로 구성된 깔끔한 결정성 모델을 가지지만, idx 103 의 CH3/CH4 는 |H(f)| 가 전 대역에서 0에 가까움 — AR 예측기가 모델할 결정성 자체가 없다는 뜻.
> - **시계열 비교**: pre-event 에서는 DRS 잔차가 원본을 따라가면서 작은 결정성 굴곡만 빼낸 모습. post-event 에서는 CH3/CH4 거대 충격이 원본·잔차 양쪽에 그대로 — DRS 가 충격을 보존(의도대로).
> - **PSD 비교**: pre-event 에서 결정성 spectral line(좁은 피크)들이 잔차에서 사라지고 broadband carpet 만 남음. post-event CH3/CH4 는 PSD 변화가 거의 없음 (제거할 결정성이 없으므로).
> - **결론**: DRS 는 **충격 이벤트 *전* 신호 4 채널 모두**에서 결정성을 깔끔히 분리한다. 그러나 **이벤트 이후 CH3/CH4 의 fallback 은 알고리즘으로 풀 수 없는 신호 한계** — 해법은 `features_utils.per_train_bands` 가 Train2 의 BP 선정을 idx ~70 부근 파일에서 수행하도록 바꾸는 것."""


def md_cell(cid: str, src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code_cell(cid: str, src: str) -> dict:
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# Idempotent: drop any pre-existing s4b-* cells, then re-insert
nb["cells"] = [c for c in nb["cells"] if not c.get("id", "").startswith("t2-s4b-")]

for i, c in enumerate(nb["cells"]):
    if c.get("id") == "t2-s4a-insight":
        insert_after = i
        break
else:
    raise SystemExit("t2-s4a-insight not found")

new_cells = [
    md_cell("t2-s4b-md", S4B_MD),
    code_cell("t2-s4b-code1", S4B_CODE1),
    code_cell("t2-s4b-code2", S4B_CODE2),
    code_cell("t2-s4b-code3", S4B_CODE3),
    code_cell("t2-s4b-code4", S4B_CODE4),
    md_cell("t2-s4b-insight", S4B_INSIGHT),
]

nb["cells"] = nb["cells"][: insert_after + 1] + new_cells + nb["cells"][insert_after + 1:]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"inserted {len(new_cells)} cells; total cells now {len(nb['cells'])}")
