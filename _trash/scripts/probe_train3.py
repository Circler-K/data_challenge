"""One-off probe to print key Train3 numbers for the analysis notebook."""
import sys
from pathlib import Path
ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

import numpy as np
import pandas as pd

from src.io_tdms import load_tdms_file, tdms_to_array, FS, CHANNEL_NAMES
from src.operation import load_operation, align_to_vibration, list_vibration_files

TR = 3
op = load_operation(TR)
files = list_vibration_files(TR)
agg = align_to_vibration(op, len(files))

print("=== Train3 Operation profile ===")
print(f"op rows:           {len(op):,}")
print(f"duration (h):      {op['Time[sec]'].iloc[-1]/3600:.3f}")
print(f"vibration files:   {len(files)}")
print(f"first / last file: {files[0].name} / {files[-1].name}")
print()

# Operational extremes
print("=== Operational extremes (whole CSV) ===")
print(f"RPM   min/max/mean: {op['Motor speed[rpm]'].min():.1f} / {op['Motor speed[rpm]'].max():.1f} / {op['Motor speed[rpm]'].mean():.1f}")
print(f"Torque min/max/mean: {op['Torque[Nm]'].min():.2f} / {op['Torque[Nm]'].max():.2f} / {op['Torque[Nm]'].mean():.2f}")
print(f"TC SP Front max:    {op['TC SP Front'].max():.2f}")
print(f"TC SP Rear  max:    {op['TC SP Rear'].max():.2f}")
print(f"Last Torque value: {op['Torque[Nm]'].iloc[-1]:.2f}")
print(f"Last RPM   value:  {op['Motor speed[rpm]'].iloc[-1]:.2f}")
print(f"Last Front T value: {op['TC SP Front'].iloc[-1]:.2f}")
print(f"Last Rear  T value: {op['TC SP Rear'].iloc[-1]:.2f}")
print()

# Endpoint RPMs
print("=== RPM at first / last vibration file ===")
print(f"first file rpm_mean: {agg['rpm_mean'].iloc[0]:.1f}")
print(f"last  file rpm_mean: {agg['rpm_mean'].iloc[-1]:.1f}")
print()

# Feature parquet summary (RMS / kurtosis × 4 channels, early vs late)
df = pd.read_parquet(ROOT / "outputs/features_utils/train3.parquet").sort_values("file_idx").reset_index(drop=True)
print(f"feature df shape: {df.shape}")
print(f"feature columns count: {len(df.columns)}")

print()
print("=== Channel time-domain summary (Train3) ===")
print(f"{'Ch':<4} {'rms_e':>8} {'rms_l':>8} {'rms_x':>6} {'kurt_e':>8} {'kurt_l':>10} {'cf_e':>6} {'cf_l':>6}")
print('-' * 70)
for ch in CHANNEL_NAMES:
    rms_e = df[f'{ch}_rms'].iloc[0]
    rms_l = df[f'{ch}_rms'].iloc[-1]
    kurt_e = df[f'{ch}_kurt'].iloc[0]
    kurt_l = df[f'{ch}_kurt'].iloc[-1]
    cf_e = df[f'{ch}_cf'].iloc[0]
    cf_l = df[f'{ch}_cf'].iloc[-1]
    print(f"{ch:<4} {rms_e:8.4f} {rms_l:8.4f} {rms_l/rms_e:6.2f} {kurt_e:8.3f} {kurt_l:10.3f} {cf_e:6.2f} {cf_l:6.2f}")

print()
print("=== Channel time-domain summary (max within run) ===")
for ch in CHANNEL_NAMES:
    rms_max_idx = int(df[f'{ch}_rms'].idxmax())
    kurt_max_idx = int(df[f'{ch}_kurt'].idxmax())
    print(f"{ch}: rms_max={df[f'{ch}_rms'].iloc[rms_max_idx]:.4f} @ file{rms_max_idx+1}  kurt_max={df[f'{ch}_kurt'].iloc[kurt_max_idx]:.1f} @ file{kurt_max_idx+1}")

print()
print("=== file 0 (early) operation row ===")
print(agg.iloc[0].to_string())

print()
print("=== file -1 (late) operation row ===")
print(agg.iloc[-1].to_string())

print()
# RPM regime detection
print("=== RPM histogram (Operation full) ===")
rpm_arr = op['Motor speed[rpm]'].to_numpy()
hist, edges = np.histogram(rpm_arr, bins=20)
for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
    if h > 0:
        print(f"  [{e0:.0f}, {e1:.0f}]: {h}")
