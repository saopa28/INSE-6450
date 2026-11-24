"""Uplift metrics (identical definitions to Milestones 2-3) and bootstrap helpers."""
import numpy as np

from .config import SEED


def gain_curve(y, t, s, auuc=False):
    o = np.argsort(-s, kind="stable"); y, t = y[o], t[o]
    nt, nc, yt, yc = np.cumsum(t), np.cumsum(1 - t), np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(yt / nt - yc / nc) * (nt + nc) if auuc else yt - yc * np.where(nc > 0, nt / nc, 0)


def qini(y, t, s, auuc=False):
    q = gain_curve(y, t, s, auuc); n = len(q)
    return float(np.mean(q - np.linspace(q[-1] / n, q[-1], n)) / n)


def uplift_at(y, t, s, k=0.10):
    top = np.argsort(-s, kind="stable")[: max(1, int(len(s) * k))]
    yt, tt = y[top], t[top]
    if tt.min() == tt.max():
        return float("nan")
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def summary(y, t, s):
    return {"qini": qini(y, t, s), "auuc": qini(y, t, s, auuc=True), "uplift@10%": uplift_at(y, t, s, 0.10)}


def paired_boot(y, t, s_new, s_old, n=100, seed=SEED):
    """95% bootstrap interval of qini(new) - qini(old) on the same resamples, and the share where new wins."""
    rng, d = np.random.default_rng(seed), np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(y), len(y))
        d[i] = qini(y[j], t[j], s_new[j]) - qini(y[j], t[j], s_old[j])
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))), float((d > 0).mean())


def qini_floor(y, t, s, size, n=50, seed=SEED):
    """Floor for a Qini check on batches of `size` rows: mean minus two standard deviations."""
    rng = np.random.default_rng(seed)
    vals = [qini(y[j], t[j], s[j]) for j in (rng.choice(len(y), size, replace=False) for _ in range(n))]
    return float(np.mean(vals) - 2 * np.std(vals)), float(np.mean(vals)), float(np.std(vals))
