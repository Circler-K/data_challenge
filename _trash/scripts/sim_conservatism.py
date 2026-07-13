"""How conservative, exactly? Derive the optimal submission fraction f* from the
official asymmetric A_RUL metric, as a function of our RELATIVE uncertainty sigma.

Decision model: our point estimate is the MEDIAN of our belief about true RUL.
  true_RUL = estimate * L,   L ~ LogNormal(0, sigma)   (median L = 1)
We submit  pred = f * estimate. The estimate cancels in the % error:
  Er% = 100 * (true - pred)/true = 100 * (1 - f/L)
so the optimal f depends ONLY on sigma (how uncertain we are), not the magnitude.
We pick f maximising the EXPECTED A_RUL over the belief distribution.

This turns "be conservative" into a number: e.g. sigma=1.0 -> submit ~45% of the
estimate. Then map each validation bearing to a confidence tier (sigma) and read
off its f*.

Run:  python scripts/sim_conservatism.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path("c:/Users/User/WorkSpace/data_challenge")
sys.path.insert(0, str(ROOT))
from src.scoring import a_rul_score  # noqa: E402

RNG = np.random.default_rng(42)
N = 200_000
FRACS = np.round(np.arange(0.05, 1.51, 0.05), 2)


def optimal_fraction(sigma):
    """Return (f*, E[score at f*], E[score at f=1.0]) for relative uncertainty sigma."""
    L = np.exp(RNG.normal(0.0, sigma, N))            # true/estimate, median 1
    best_f, best_s = 1.0, -1.0
    s_at_1 = None
    for f in FRACS:
        er = 100.0 * (1.0 - f / L)                    # Er% per draw
        s = float(np.mean(a_rul_score(np.ones(N), 1.0 - er / 100.0)))  # act=1, pred=1-er/100
        if abs(f - 1.0) < 1e-9:
            s_at_1 = s
        if s > best_s:
            best_s, best_f = s, f
    return best_f, best_s, s_at_1


def main():
    print("=== Optimal submission fraction f* vs uncertainty sigma ===")
    print("(submit pred = f* x your central estimate; estimate cancels, depends only on sigma)\n")
    print(f"{'sigma':>6} {'~spread':>18} {'f*':>6} {'E[score]@f*':>12} {'E[score]@100%':>14} {'gain':>7}")
    table = {}
    for sigma in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.4, 1.8):
        f, s, s1 = optimal_fraction(sigma)
        table[sigma] = f
        lo, hi = np.exp(-sigma), np.exp(sigma)
        spread = f"x{lo:.2f}..x{hi:.2f}"
        print(f"{sigma:>6.1f} {spread:>18} {f:>6.2f} {s:>12.3f} {s1:>14.3f} {s-s1:>+7.3f}")

    # ---- per-bearing application ----
    # central estimate (h): use the degradation-RATE RUL from diagnose_validation;
    # flat bearings get a pre-knee central, NOT the cap (a flat segment that may be
    # near failure must not be assumed long).
    print("\n=== per-bearing recommendation ===")
    print("tier sigma chosen from signal quality (slope clarity, HI level, method agreement):\n")
    bearings = [
        # HONEST central (pre-knee/low-HI => LONG, not short); ONE f* shrink, moderate sigma.
        # name,   central_h, tier,    sigma, reason
        ("Test1",  15.0, "MED",   0.6, "flat + LOW HI = pre-knee => long life; uncertainty is WHEN the knee hits"),
        ("Test2",  12.0, "MED",   0.8, "mid HI, flat/neg slope: likely plateau, long-ish but less sure"),
        ("Test3",   9.9, "MED",   0.6, "mild steady rise, mid HI: degradation visible"),
        ("Test4",   9.5, "MED",   0.7, "steep slope (0.24/h) but low HI: active, magnitude uncertain"),
        ("Test5",   8.0, "LOW",   0.9, "mild rise, methods disagree: weakest signal"),
        ("Test6",   1.2, "HIGH",  0.4, "high HI (1.74) + steep rise (0.28/h): clearly late-stage"),
    ]

    def fstar(sigma):
        # nearest tabulated sigma
        ks = sorted(table)
        k = min(ks, key=lambda x: abs(x - sigma))
        return table[k]

    print(f"{'bearing':>8} {'tier':>5} {'sigma':>6} {'central_h':>10} {'f*':>6} "
          f"{'submit_h':>9} {'submit_sec':>11}  reason")
    subs = {}
    for name, c_h, tier, sig, why in bearings:
        f = fstar(sig)
        sub_h = c_h * f
        sub_s = int(round(sub_h * 3600))
        subs[name] = sub_s
        print(f"{name:>8} {tier:>5} {sig:>6.1f} {c_h:>10.1f} {f:>6.2f} "
              f"{sub_h:>9.1f} {sub_s:>11d}  {why}")
    print("\nproposed submission (sec):", subs)


if __name__ == "__main__":
    main()
