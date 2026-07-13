"""Official KSPHM-KIMM 2026 RUL scoring metric (A_RUL).

The challenge evaluates a *single* RUL prediction per bearing, taken at the
last measurement timestamp of the validation segment, in seconds.

Error (percent):
    Er_i = 100 * (ActRUL_i - hatRUL_i) / ActRUL_i

    Er_i <= 0  <=>  hatRUL_i >= ActRUL_i   (OVER-prediction: predicted more life)
    Er_i  > 0  <=>  hatRUL_i <  ActRUL_i   (UNDER-prediction: predicted less life)

Score (per bearing) — OFFICIAL (organizer spec, /30 for over-prediction):
    A_RUL = exp(-ln(0.5) * Er/30)   if Er <= 0   (over-prediction, /30)
            exp(+ln(0.5) * Er/50)   if Er  > 0   (under-prediction, gentle /50)

Properties (sanity anchors):
    Er =   0  -> 1.0   (perfect)
    Er = -30  -> 0.5   (30% over-prediction halves the score)
    Er = +50  -> 0.5   (50% under-prediction halves the score)
    Er -> -inf -> 0    (large over-prediction punished fast)

=> The metric rewards CONSERVATIVE (slightly short) RUL predictions. Any model
   tuning should bias toward under-prediction, never over-prediction.

Final score = mean of A_RUL over all bearings.
"""
from __future__ import annotations

import numpy as np

LN_HALF = np.log(0.5)  # = -0.6931... (negative)


def error_pct(act_rul, pred_rul):
    """Er_i in percent. act_rul must be > 0."""
    act = np.asarray(act_rul, dtype=np.float64)
    pred = np.asarray(pred_rul, dtype=np.float64)
    return 100.0 * (act - pred) / act


def a_rul_score(act_rul, pred_rul):
    """Per-bearing A_RUL score in [0, 1]. Accepts scalars or arrays."""
    er = error_pct(act_rul, pred_rul)
    # Er <= 0 (over-prediction): exp(-ln(0.5) * Er / 30)  [OFFICIAL]
    # Er  > 0 (under-prediction): exp(+ln(0.5) * Er / 50)
    over = np.exp(-LN_HALF * er / 30.0)
    under = np.exp(LN_HALF * er / 50.0)
    return np.where(er <= 0.0, over, under)


def final_score(act_rul, pred_rul):
    """Mean A_RUL over all bearings — the number the challenge ranks on."""
    return float(np.mean(a_rul_score(act_rul, pred_rul)))


if __name__ == "__main__":
    # Anchor checks
    for er_target, act in [(0, 1000), (-20, 1000), (50, 1000)]:
        pred = act * (1 - er_target / 100.0)
        print(f"Er={er_target:+4d}%  act={act}  pred={pred:.0f}  "
              f"A_RUL={a_rul_score(act, pred):.4f}")
