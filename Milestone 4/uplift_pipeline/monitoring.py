"""Data-quality and drift monitoring (Milestone 3 Section 3) and the update trigger (Section 4.1)."""
import numpy as np

from .config import FEATS, CRITICAL, PSI_WARN, PSI_BLOCK, NULL_MAX, NULL_MAX_CRITICAL, OOR_MAX
from .metrics import qini

_CRIT = [FEATS.index(c) for c in CRITICAL]


class Monitor:
    """Checks a batch against a reference sample: schema, nulls, out-of-range rows and per-feature PSI."""

    def __init__(self, X_ref, bins=10):
        self.set_reference(X_ref, bins)

    def set_reference(self, X_ref, bins=10):
        X_ref = X_ref[np.random.default_rng(0).choice(len(X_ref), min(len(X_ref), 500_000), replace=False)]
        self.lo, self.hi = np.nanmin(X_ref, 0), np.nanmax(X_ref, 0)
        self.edges, self.ref = [], []
        for i in range(X_ref.shape[1]):
            e = np.unique(np.nanquantile(X_ref[:, i], np.linspace(0, 1, bins + 1)))
            e[0], e[-1] = -np.inf, np.inf
            self.edges.append(e)
            self.ref.append(np.clip(np.histogram(X_ref[:, i], e)[0] / len(X_ref), 1e-6, None))

    def psi(self, X):
        out = {}
        for i, f in enumerate(FEATS):
            x = X[:, i][~np.isnan(X[:, i])]
            cur = np.clip(np.histogram(x, self.edges[i])[0] / max(len(x), 1), 1e-6, None)
            out[f] = float(np.sum((cur - self.ref[i]) * np.log(cur / self.ref[i])))
        return out

    def check(self, X):
        psis = self.psi(X)
        nulls = np.isnan(X).mean(0)
        with np.errstate(invalid="ignore"):
            oor = float(((X < self.lo) | (X > self.hi)).any(1).mean())
        r = dict(schema_ok=bool(X.ndim == 2 and X.shape[1] == len(FEATS)),
                 psi_max=max(psis.values()), psi_feature=max(psis, key=psis.get),
                 null_max=float(nulls.max()), null_max_critical=float(nulls[_CRIT].max()), oor_rate=oor)
        alerts = []
        if not r["schema_ok"]:
            alerts.append("BLOCK: schema")
        if r["null_max"] > NULL_MAX or r["null_max_critical"] > NULL_MAX_CRITICAL:
            alerts.append("BLOCK: null rate")
        if oor > OOR_MAX:
            alerts.append("BLOCK: out-of-range rows")
        if r["psi_max"] >= PSI_BLOCK:
            alerts.append("BLOCK: PSI >= 0.25 -> run Qini check")
        elif r["psi_max"] >= PSI_WARN:
            alerts.append("WARN: PSI 0.1-0.25")
        r["alerts"] = alerts
        r["status"] = "BLOCK" if any(a.startswith("BLOCK") for a in alerts) else ("WARN" if alerts else "OK")
        return r


def qini_check(model, X, t, y, floor, windows=2):
    """Milestone 3 retraining trigger: Qini below the floor in `windows` consecutive windows.
    The labelled holdout batch is cut into consecutive windows; returns (triggered, per-window Qini)."""
    s = model.score(X)
    q = [qini(y[w], t[w], s[w]) for w in np.array_split(np.arange(len(y)), windows)]
    return bool(all(v < floor for v in q)), q
