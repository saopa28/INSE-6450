"""Data loading, the Milestone 2 split, and drift injection."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_URL, FEATS, RAW, TARGET, SEED


def load_data(path=DATA_URL, frac=1.0, seed=SEED, chunksize=1_000_000):
    """Read the engineered Criteo file in chunks; frac < 1 keeps a random share of rows (for quick runs)."""
    cols = FEATS + ["treatment", TARGET]
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, compression="infer", usecols=cols,
                                          dtype={c: "float32" for c in FEATS}, chunksize=chunksize)):
        if frac < 1.0:
            chunk = chunk.sample(frac=frac, random_state=seed + i)
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def split(df, seed=SEED):
    """Same 70/15/15 split as Milestone 2, stratified on treatment x conversion."""
    strata = lambda d: d.treatment.astype(int) * 2 + d[TARGET].astype(int)
    tr, rest = train_test_split(df, test_size=0.30, stratify=strata(df), random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=strata(rest), random_state=seed)
    arr = lambda d: (np.ascontiguousarray(d[FEATS].to_numpy(np.float32)),
                     d.treatment.to_numpy(np.int8), d[TARGET].to_numpy(np.int8))
    return {"train": arr(tr), "val": arr(va), "test": arr(te)}


_IDX = {c: i for i, c in enumerate(FEATS)}


def rebuild(X):
    """Recompute the six engineered features from the raw columns (in place) and return X."""
    f4, f9, f11 = X[:, _IDX["f4"]], X[:, _IDX["f9"]], X[:, _IDX["f11"]]
    X[:, _IDX["f4_x_f9"]], X[:, _IDX["f4_x_f11"]], X[:, _IDX["f9_x_f11"]] = f4 * f9, f4 * f11, f9 * f11
    X[:, _IDX["f4_squared"]], X[:, _IDX["f9_squared"]], X[:, _IDX["f11_squared"]] = f4 ** 2, f9 ** 2, f11 ** 2
    return X


def raw_sd(X):
    """Per-feature standard deviation of the 12 raw features (reference for drift sizes)."""
    return X[:, :len(RAW)].std(axis=0)


def shift(X, sd, k, cols=None):
    """Return a copy of X with the given raw columns shifted by k standard deviations.
    k == 0 returns X unchanged (no recomputation, so clean data stays bit-identical)."""
    if not k:
        return X
    X = X.copy()
    for c in (cols or RAW):
        X[:, _IDX[c]] += k * sd[_IDX[c]]
    return rebuild(X)
