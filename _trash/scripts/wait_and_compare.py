"""Poll until denoise feature CSVs are complete, then print the HI-quality ranking.
Used to monitor the detached per-bearing denoise jobs and emit the final table.

    python scripts/wait_and_compare.py --tag full  --denoisers none,wav_univ,... --bearings 1,2,3,4
    python scripts/wait_and_compare.py --tag probe --denoisers none,vmd,emd      --bearings 3
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
OUT = ROOT / "outputs" / "ot_features" / "denoise"
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--denoisers", required=True)
    ap.add_argument("--bearings", required=True)
    ap.add_argument("--timeout", type=int, default=18000)
    args = ap.parse_args()
    names = args.denoisers.split(",")
    bearings = [int(b) for b in args.bearings.split(",")]
    need = [OUT / args.tag / n / f"Train{tr}.csv" for n in names for tr in bearings]

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        missing = [p for p in need if not p.exists()]
        if not missing:
            break
        print(f"[{int(time.time()-t0)}s] waiting, {len(need)-len(missing)}/{len(need)} CSVs ready", flush=True)
        time.sleep(30)

    print(f"\n[done waiting after {int(time.time()-t0)}s] running compare:\n", flush=True)
    subprocess.run([PY, str(ROOT / "scripts" / "denoise_compare.py"),
                    "--compare-only", "--tag", args.tag,
                    "--denoisers", args.denoisers, "--bearings", args.bearings])


if __name__ == "__main__":
    main()
