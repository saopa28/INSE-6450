"""Compact stress tests from Milestone 3 Section 1 (input corruptions; no retraining)."""
import numpy as np
import pandas as pd

from .config import RAW, SEED
from .data import rebuild
from .metrics import summary


def stress_tests(model, X, t, y, monitor, seed=SEED):
    rng = np.random.default_rng(seed)
    R = X[:, :len(RAW)]
    sd, mu = R.std(0), np.nanmean(R, 0)

    def build(raw):
        Z = X.copy(); Z[:, :len(RAW)] = raw
        return rebuild(Z)

    masked = R.copy(); masked[rng.random(R.shape) < 0.10] = np.nan
    lost = R.copy(); lost[:, 9] = mu[9]
    cases = {"Clean": X,
             "Noise 0.5 sd": build(R + rng.normal(0, 0.5, R.shape).astype(np.float32) * sd),
             "Rounded values": build(np.round(R)),
             "Masking 10%": build(masked),
             "Feature loss f9": build(lost),
             "OOD shift +3 sd": build(R + 3 * sd)}
    rows = []
    for name, Z in cases.items():
        s = model.score(Z)
        rows.append({"scenario": name, **summary(y, t, s), "preflight": monitor.check(Z)["status"]})
    df = pd.DataFrame(rows).set_index("scenario")
    df["qini_%_of_clean"] = (100 * df.qini / df.qini.iloc[0]).round(1)
    return df
