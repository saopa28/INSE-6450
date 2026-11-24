"""X-learner (Milestone 2 architecture) with full training, warm-start updates, and versioned save/load."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import xgboost as xgb

from .config import CLF_DEFAULT, REG_DEFAULT, FEATS, SEED, V1_LOCAL_DIR, V1_URL, OUT_DIR

PARTS = ("mu1", "mu0", "tau1", "tau0")


def n_trees(m):
    """Trees actually used for prediction (best iteration + 1 when early stopping was used)."""
    best = getattr(m, "best_iteration", None)
    if best is not None:
        return int(best) + 1
    return int(m.get_booster().num_boosted_rounds())


def _fit_one(cls, cfg, X, y, Xv, yv, base=None, new_trees=None, lr=None, early_stop=None):
    """Fit one booster. With `base`, continue boosting from it (warm start): its trees are kept
    unchanged and at most `new_trees` trees are added, with their own learning rate."""
    c = dict(cfg)
    init = None
    if base is not None:
        c["n_estimators"] = int(new_trees)
        if lr is not None:
            c["learning_rate"] = lr
        if early_stop is not None:
            c["early_stopping_rounds"] = early_stop
        init = base.get_booster()[: n_trees(base)]       # drop trees beyond the old best iteration
    m = cls(**c)
    m.fit(X, y, eval_set=[(Xv, yv)], verbose=False, xgb_model=init)
    return m


def _pred(m, X):
    return m.get_booster().inplace_predict(X, iteration_range=(0, n_trees(m)))


class XLearner:
    """Four boosters: mu1/mu0 (conversion per arm) and tau1/tau0 (uplift); score = g*tau0 + (1-g)*tau1."""

    def __init__(self, mu1, mu0, tau1, tau0, g, clf=None, reg=None, meta=None):
        self.mu1, self.mu0, self.tau1, self.tau0, self.g = mu1, mu0, tau1, tau0, float(g)
        self.clf, self.reg = dict(clf or CLF_DEFAULT), dict(reg or REG_DEFAULT)
        self.meta = dict(meta or {})

    # ---------- training ----------
    @classmethod
    def fit(cls, X, t, y, Xv, tv, yv, clf=None, reg=None, init=None, new_trees=200, lr=0.05, early_stop=20):
        """Full training (init=None) or warm-start update of every booster from `init`."""
        clf, reg = dict(clf or (init.clf if init else CLF_DEFAULT)), dict(reg or (init.reg if init else REG_DEFAULT))
        kw = dict(new_trees=new_trees, lr=lr, early_stop=early_stop)
        T, C, Tv, Cv = t == 1, t == 0, tv == 1, tv == 0
        b = (lambda k: getattr(init, k)) if init is not None else (lambda k: None)
        mu1 = _fit_one(xgb.XGBClassifier, clf, X[T], y[T], Xv[Tv], yv[Tv], b("mu1"), **kw)
        mu0 = _fit_one(xgb.XGBClassifier, clf, X[C], y[C], Xv[Cv], yv[Cv], b("mu0"), **kw)
        d1, d1v = y[T] - _pred(mu0, X[T]), yv[Tv] - _pred(mu0, Xv[Tv])
        d0, d0v = _pred(mu1, X[C]) - y[C], _pred(mu1, Xv[Cv]) - yv[Cv]
        tau1 = _fit_one(xgb.XGBRegressor, reg, X[T], d1, Xv[Tv], d1v, b("tau1"), **kw)
        tau0 = _fit_one(xgb.XGBRegressor, reg, X[C], d0, Xv[Cv], d0v, b("tau0"), **kw)
        return cls(mu1, mu0, tau1, tau0, float(t.mean()), clf, reg)

    # ---------- inference ----------
    def score(self, X):
        """Uplift score. Uses only the two stage-2 models."""
        return self.g * _pred(self.tau0, X) + (1 - self.g) * _pred(self.tau1, X)

    def uncertainty(self, X):
        """Disagreement between the two uplift estimators, |tau1 - tau0|: high where the model is unsure."""
        return np.abs(_pred(self.tau1, X) - _pred(self.tau0, X))

    def conversion_proba(self, X, treated=True):
        return _pred(self.mu1 if treated else self.mu0, X)

    def trees(self):
        return {k: n_trees(getattr(self, k)) for k in PARTS}

    def size_mb(self, scoring_only=True):
        parts = ("tau1", "tau0") if scoring_only else PARTS
        return sum(len(getattr(self, k).get_booster()[: n_trees(getattr(self, k))].save_raw(raw_format="ubj"))
                   for k in parts) / 1e6

    # ---------- versioning ----------
    def save(self, directory, version, parent=None, **extra):
        """Write the four boosters (JSON) and a manifest with hashes, lineage, config and metrics."""
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for k in PARTS:
            p = d / f"{k}.json"
            getattr(self, k).save_model(str(p))
            hashes[k] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        manifest = dict(version=version, parent=parent, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                        seed=SEED, features=FEATS, g=self.g, clf=self.clf, reg=self.reg,
                        trees=self.trees(), scoring_size_mb=round(self.size_mb(), 4),
                        artifact_sha256=hashes, xgboost_version=xgb.__version__, **extra)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        self.meta = manifest
        return manifest

    @classmethod
    def load(cls, directory):
        """Load a version folder written by `save`, or the Milestone 2 folder (metadata.json)."""
        d = Path(directory)
        meta_file = d / "manifest.json" if (d / "manifest.json").exists() else d / "metadata.json"
        meta = json.loads(meta_file.read_text())
        parts = {}
        for k, c in zip(PARTS, (xgb.XGBClassifier, xgb.XGBClassifier, xgb.XGBRegressor, xgb.XGBRegressor)):
            m = c(); m.load_model(str(d / f"{k}.json")); parts[k] = m
        return cls(**parts, g=meta["g"], clf=meta.get("clf"), reg=meta.get("reg"), meta=meta)


def load_v1(cache_dir=None):
    """The Milestone 2 X-learner: from the local repo if available, otherwise downloaded from GitHub."""
    if V1_LOCAL_DIR.exists() and all((V1_LOCAL_DIR / f"{k}.json").exists() for k in PARTS):
        return XLearner.load(V1_LOCAL_DIR)
    d = Path(cache_dir or OUT_DIR / "models" / "v1_milestone2"); d.mkdir(parents=True, exist_ok=True)
    for name in list(PARTS) + ["metadata"]:
        if not (d / f"{name}.json").exists():
            urllib.request.urlretrieve(f"{V1_URL}/{name}.json", d / f"{name}.json")
    return XLearner.load(d)
