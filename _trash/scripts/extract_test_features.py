"""Extract vibration features from Test/Test1-6 TDMS files.

Test data has no Operation CSV — substitute Train1-4 *mean* operation values
so the LightGBM model sees the same schema it was trained on. Computed once
from train{1..4}.parquet on first run and cached in OP_MEANS dict below.

  - RPM (= 849.07) drives the envelope band scaling
  - BP filter uses fallback [1000, 10000] Hz (no per-Test kurtogram run here)
  - Operation columns are filled with Train1-4 grand mean — broadcast same
    value to every Test row. Not unit-specific, but matches training schema.

Output: outputs/features_utils/test_all.parquet — same schema as
train{1..4}.parquet, with train_id ∈ {101..106} (offset to avoid clash).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(ROOT / "utils"))

import numpy as np
import pandas as pd

from src.io_tdms import CHANNEL_NAMES, load_tdms_file, tdms_to_array
from src.features_utils import channel_features, KURT_FALLBACK_BAND

TEST_ROOT = ROOT / "Test" / "Test"   # canonical inner folder
OUT_DIR = ROOT / "outputs" / "features_utils"
DEFAULT_BAND = KURT_FALLBACK_BAND  # (1000, 10000) Hz

# Train1-4 grand-mean operation values (substitute for missing Test operation CSV)
OP_MEANS = dict(
    rpm_mean=849.07,
    rpm_std=3.52,
    torque_mean=-6.20,
    torque_min=-6.81,
    torque_std=0.41,
    tcf_mean=83.64,
    tcr_mean=95.63,
    tcf_max=84.15,
    tcr_max=96.12,
)
DEFAULT_RPM = OP_MEANS["rpm_mean"]  # used for envelope band scaling


def find_test_units() -> dict[str, list[Path]]:
    """Locate TDMS files per unit, handling Test1/Test6 nested layout."""
    units: dict[str, list[Path]] = {}
    for n in range(1, 7):
        unit_dir = TEST_ROOT / f"Test{n}"
        nested = unit_dir / f"Test{n}"
        if nested.is_dir():
            files = sorted(nested.glob("*.tdms"))
        else:
            files = sorted(unit_dir.glob("*.tdms"))
        if files:
            units[f"Test{n}"] = files
    return units


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = find_test_units()
    print(f"Found {len(units)} test units:")
    for name, files in units.items():
        print(f"  {name}: {len(files)} TDMS files")

    parts = []
    t0 = time.time()
    for unit_name, files in units.items():
        unit_id = int(unit_name.replace("Test", ""))
        rows = []
        for i, path in enumerate(files, start=1):
            arr = tdms_to_array(load_tdms_file(path))
            row = dict(
                train_id=100 + unit_id,                # 101..106
                file_idx=i,
                file_name=path.name,
                t_start_sec=(i - 1) * 600,            # 10-min capture period
                time_to_eol_sec=np.nan,
                life_frac=np.nan,
            )
            for j, ch in enumerate(CHANNEL_NAMES):
                feats = channel_features(
                    arr[j], DEFAULT_RPM,
                    DEFAULT_BAND[0], DEFAULT_BAND[1],
                )
                for k, v in feats.items():
                    row[f"{ch}_{k}"] = v
            rows.append(row)
            if i % 10 == 0 or i == len(files):
                print(f"    {unit_name} {i}/{len(files)}  "
                      f"({time.time() - t0:.0f}s)")
        df = pd.DataFrame(rows)
        for col, val in OP_MEANS.items():
            df[col] = val
        parts.append(df)

    all_df = pd.concat(parts, ignore_index=True)
    out = OUT_DIR / "test_all.parquet"
    all_df.to_parquet(out, index=False)
    print(f"\nSaved {out}  shape={all_df.shape}")


if __name__ == "__main__":
    main()
