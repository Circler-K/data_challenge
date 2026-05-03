"""Operation CSV (Torque/RPM/Temperature) loading and alignment to TDMS files.

Operation CSVs are 0.1 Hz (10-second period). Vibration TDMS files capture 1
minute every 10 minutes, so for the i-th file (1-based) we aggregate the
operation rows whose ``Time[sec]`` falls within
``[(i-1)*600, (i-1)*600 + 60]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("c:/Users/User/WorkSpace/data_challenge/Train")

OP_COLS = (
    "Time[sec]",
    "Torque[Nm]",
    "Motor speed[rpm]",
    "TC SP Front",
    "TC SP Rear",
)


def load_operation(train_id: int, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load Train{id}_Operation.csv as a clean DataFrame.

    The original headers contain stray Korean degree-sign bytes; we ignore
    them by reading with latin1 and renaming the temperature columns.
    """
    path = root / f"Train{train_id}_Operation.csv"
    df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    rename = {}
    for col in df.columns:
        if "TC SP Front" in col:
            rename[col] = "TC SP Front"
        elif "TC SP Rear" in col:
            rename[col] = "TC SP Rear"
    df = df.rename(columns=rename)
    return df[list(OP_COLS)].copy()


def align_to_vibration(
    df_op: pd.DataFrame,
    n_files: int,
    period_min: int = 10,
    capture_sec: int = 60,
) -> pd.DataFrame:
    """Aggregate operation rows for each vibration file's capture window.

    Returns one row per vibration file with rpm/torque/temp summary stats.
    """
    rows = []
    t = df_op["Time[sec]"].to_numpy()
    rpm = df_op["Motor speed[rpm]"].to_numpy()
    tq = df_op["Torque[Nm]"].to_numpy()
    tcf = df_op["TC SP Front"].to_numpy()
    tcr = df_op["TC SP Rear"].to_numpy()
    for i in range(1, n_files + 1):
        start = (i - 1) * period_min * 60
        end = start + capture_sec
        mask = (t >= start) & (t < end + 10)  # +10 to grab one extra row
        if not mask.any():
            mask = (t >= start - 10) & (t <= end + 10)
        rows.append(
            dict(
                file_idx=i,
                t_start_sec=start,
                rpm_mean=float(rpm[mask].mean()) if mask.any() else np.nan,
                rpm_std=float(rpm[mask].std()) if mask.any() else np.nan,
                torque_mean=float(tq[mask].mean()) if mask.any() else np.nan,
                torque_min=float(tq[mask].min()) if mask.any() else np.nan,
                torque_std=float(tq[mask].std()) if mask.any() else np.nan,
                tcf_mean=float(tcf[mask].mean()) if mask.any() else np.nan,
                tcr_mean=float(tcr[mask].mean()) if mask.any() else np.nan,
                tcf_max=float(tcf[mask].max()) if mask.any() else np.nan,
                tcr_max=float(tcr[mask].max()) if mask.any() else np.nan,
            )
        )
    return pd.DataFrame(rows)


def list_vibration_files(train_id: int, root: Path = DATA_ROOT) -> list[Path]:
    """Sorted list of TDMS files for a Train."""
    folder = root / f"Train{train_id}_Vibration"
    return sorted(folder.glob("*.tdms"))


if __name__ == "__main__":
    for tr in [1, 2, 3, 4]:
        df = load_operation(tr)
        files = list_vibration_files(tr)
        agg = align_to_vibration(df, len(files))
        print(
            f"Train{tr}: op_rows={len(df)}  duration={df['Time[sec]'].iloc[-1]/3600:.2f}h  "
            f"vib_files={len(files)}  agg_rows={len(agg)}"
        )
        print(agg.head(2).to_string(index=False))
        print(agg.tail(2).to_string(index=False))
        print()
