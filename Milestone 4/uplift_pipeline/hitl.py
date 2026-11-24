"""Human-in-the-loop and active learning (simulated).

In this problem a "label" is not a human annotation: the outcome of one user is only observed when the
user is randomly assigned to treatment or control and then converts (or not). The oracle is therefore a
randomized experiment run on the users we choose to query; in the simulation it reveals the logged
(treatment, conversion) of those users, which were randomized in the Criteo data. Querying by x does not
break randomization, because treatment is still assigned at random given x.

The human reviewer is the campaign analyst who approves monitoring alerts and model promotions
(Milestone 3, Section 4.1: "promotion only after review"). The simulated reviewer applies the written
rules; with interactive=True it asks at the keyboard instead.
"""
import numpy as np
import pandas as pd

from .config import SEED, CONTINUAL
from .metrics import qini, uplift_at
from .model import XLearner


class SimulatedOracle:
    """Reveals (treatment, conversion) for queried rows of an unlabelled pool and counts the cost."""

    def __init__(self, t, y):
        self._t, self._y = t, y
        self.queried = 0

    def query(self, idx):
        self.queried += len(idx)
        return self._t[idx], self._y[idx]


class SimulatedReviewer:
    """Approves or rejects pipeline decisions; every decision is logged with its reason."""

    def __init__(self, interactive=False):
        self.interactive, self.log = interactive, []

    def _ask(self, question, default):
        if not self.interactive:
            return default
        ans = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        return default if not ans else ans.startswith("y")

    def review_alert(self, monitor_report, qini_triggered):
        """An alert opens an update only if the Qini check confirms that ranking quality fell
        (Milestone 3: PSI alone over-alerts; 0.5 sd on f9 gave PSI 13 with no ranking loss)."""
        default = bool(qini_triggered)
        ok = self._ask(f"Monitor status {monitor_report['status']} (max PSI {monitor_report['psi_max']:.2f} on "
                       f"{monitor_report['psi_feature']}), Qini trigger {qini_triggered}. Start an update?", default)
        self.log.append(dict(decision="alert", approved=ok, status=monitor_report["status"],
                             psi_max=monitor_report["psi_max"], qini_triggered=qini_triggered))
        return ok

    def review_promotion(self, ci, version):
        """Promote only if the lower end of the paired bootstrap interval of the Qini gain is above zero."""
        default = ci[0] > 0
        ok = self._ask(f"Candidate {version}: Qini gain 95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}]. Promote?", default)
        self.log.append(dict(decision="promotion", approved=ok, version=version, ci_lo=ci[0], ci_hi=ci[1]))
        return ok


def select(model, X_pool, available, budget, strategy, rng, hybrid_share=0.7):
    """Pick `budget` rows of the pool to label.
    uncertainty: largest |tau1 - tau0| (the two uplift estimators disagree most);
    random: uniform; hybrid: 70% uncertainty, 30% random (keeps coverage of the whole population)."""
    avail = np.flatnonzero(available)
    budget = min(budget, len(avail))
    if strategy == "random":
        return rng.choice(avail, budget, replace=False)
    u = model.uncertainty(X_pool[avail])
    if strategy == "uncertainty":
        return avail[np.argpartition(-u, budget - 1)[:budget]]
    if strategy == "hybrid":
        k = int(hybrid_share * budget)
        top = avail[np.argpartition(-u, k - 1)[:k]] if k else np.array([], int)
        rest = np.setdiff1d(avail, top)
        return np.concatenate([top, rng.choice(rest, budget - k, replace=False)])
    raise ValueError(strategy)


def active_learning(model, pool, val, test, rounds=4, budget=50_000, strategies=("uncertainty", "random", "hybrid"),
                    cfg=None, seed=SEED, log=print):
    """Simulated active learning: each round the deployed model chooses `budget` users from the pool,
    the oracle reveals their outcome, and the model is updated by warm-start boosting on all labels so far.
    Every strategy starts from the same model and has the same label budget; Qini is measured on `test`."""
    cfg = {**CONTINUAL, **(cfg or {})}
    Xp, tp, yp = pool
    Xte, tte, yte = test
    s0 = model.score(Xte)
    base = dict(round=0, labels=0, qini=qini(yte, tte, s0), uplift10=uplift_at(yte, tte, s0))
    rows = []
    for strat in strategies:
        rng, oracle = np.random.default_rng(seed), SimulatedOracle(tp, yp)
        avail = np.ones(len(yp), bool)
        cur, lab = model, []
        rows.append(dict(strategy=strat, **base))
        for r in range(1, rounds + 1):
            idx = select(cur, Xp, avail, budget, strat, rng)
            avail[idx] = False
            t_q, y_q = oracle.query(idx)
            lab.append(idx)
            L = np.concatenate(lab)
            cur = XLearner.fit(Xp[L], tp[L], yp[L], *val, init=model, new_trees=cfg["new_trees"],
                               lr=cfg["warm_lr"], early_stop=cfg["warm_early_stop"])
            s = cur.score(Xte)
            rows.append(dict(strategy=strat, round=r, labels=oracle.queried, qini=qini(yte, tte, s),
                             uplift10=uplift_at(yte, tte, s), treated_share=float(t_q.mean()),
                             conv_rate=float(y_q.mean())))
            log(f"  {strat:12s} round {r}: {oracle.queried:,} labels | Qini {rows[-1]['qini']:+.6f} | "
                f"conversion rate of queried users {rows[-1]['conv_rate']:.4f}")
    return pd.DataFrame(rows)


def plot_active_learning(df, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for s, g in df.groupby("strategy", sort=False):
        ax.plot(g.labels, g.qini, "o-", label=s)
    ax.set(xlabel="labels queried", ylabel="Qini on test data", title="Active learning: Qini vs label budget")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150)
    return fig
