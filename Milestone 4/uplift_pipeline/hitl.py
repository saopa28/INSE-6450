"""Human-in-the-loop and active learning (Milestone 4, Section 2).

In this problem a "label" is not a human annotation: a user's outcome is only observed when the user is
randomly assigned to treatment or control and then converts (or not). Active learning therefore chooses
which users enter a randomized experiment; in the simulation the oracle reveals the logged (treatment,
conversion) of those users, which were randomized in the Criteo data. Choosing users by x does not break
randomization, because treatment is still assigned at random given x.

Humans enter in two places:
- the reviewer approves monitoring alerts and model promotions (SimulatedReviewer, used by the demo), and
- annotators correct labels that tracking got wrong, such as bot conversions (label_curation simulation).

Every update follows the continual-learning strategy of Section 1: retrain from scratch on the new labelled
users plus a 200,000-row replay buffer, and promote only if the paired bootstrap gate on random-holdout
users passes (lower bound of the Qini gain above zero).
"""
import time

import numpy as np
import pandas as pd

from .config import SEED, RAW, CONTINUAL
from .data import shift
from .metrics import qini, uplift_at, paired_boot
from .model import XLearner

# Section 2.3 / 2.4 settings
AL = dict(init_frac=0.05, batch_frac=0.05, rounds=5, threshold=0.001, w_unc=0.5, w_margin=0.3, w_div=0.2,
          n_clusters=50, gate_rows=250_000, gate_boot=100, buffer_size=CONTINUAL["buffer_size"],
          stability_rows=100_000)
CURATION = dict(subset_frac=0.30, fake_share=0.40, review_share=0.15, cycles=3, hit_rate=0.95, false_flag=0.02)


# ------------------------------------------------------------------ humans and oracle
class SimulatedOracle:
    """Reveals (treatment, conversion) for queried rows of an unlabelled pool and counts the cost."""

    def __init__(self, t, y):
        self._t, self._y = t, y
        self.queried = 0

    def query(self, idx):
        self.queried += len(idx)
        return self._t[idx], self._y[idx]


class SimulatedReviewer:
    """Approves or rejects pipeline decisions; every decision is logged with its reason.
    With interactive=True the question is asked at the keyboard instead of by rule."""

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


class SimulatedAnnotator:
    """Reviews conversions and flags the fake ones: detects a fake with probability hit_rate and wrongly
    flags a real conversion with probability false_flag."""

    def __init__(self, is_fake, hit_rate=0.95, false_flag=0.02, seed=SEED):
        self.is_fake, self.hit, self.false = is_fake, hit_rate, false_flag
        self.seed, self.reviewed = seed, 0

    def flag(self, idx, cycle):
        self.reviewed += len(idx)
        p = np.where(self.is_fake[idx], self.hit, self.false)
        return idx[np.random.default_rng(self.seed + cycle).random(len(idx)) < p]


# ------------------------------------------------------------------ query strategy (Section 2.3)
def fit_clusters(X, n_clusters=50, seed=SEED):
    """Fit feature-space clusters on the standardized raw features; returns a function that assigns clusters.
    Used for the diversity part of the query strategy and for targeting label review."""
    from sklearn.cluster import MiniBatchKMeans
    R = X[:, :len(RAW)]
    mu, sd = R.mean(0), R.std(0)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3).fit((R[::20] - mu) / sd)
    return lambda Z: km.predict((Z[:, :len(RAW)] - mu) / sd)


def clusters(X, n_clusters=50, seed=SEED):
    """Cluster of every row of X (clusters fitted on X itself)."""
    return fit_clusters(X, n_clusters, seed)(X)


def informativeness(model, X_pool, cl_pool, cl_labeled, n_clusters=50, threshold=AL["threshold"],
                    w=(AL["w_unc"], AL["w_margin"], AL["w_div"])):
    """Hybrid score (higher = more useful to query), each part as a percentile rank:
    uncertainty |tau0 - tau1|, margin (closeness of the uplift to the profitability threshold),
    and diversity (clusters that are rare among the users labelled so far)."""
    unc, u = model.uncertainty(X_pool), model.score(X_pool)
    share = np.bincount(cl_labeled, minlength=n_clusters) / max(len(cl_labeled), 1)
    r = lambda v: pd.Series(v).rank(pct=True).values
    return w[0] * r(unc) + w[1] * r(-np.abs(u - threshold)) + w[2] * r(-share[cl_pool])


def select(model, X_pool, budget, X_labeled=None, n_clusters=AL["n_clusters"], threshold=AL["threshold"],
           w=(AL["w_unc"], AL["w_margin"], AL["w_div"]), seed=SEED):
    """Pick the `budget` most informative users of X_pool for the next experiment (hybrid query strategy).
    X_labeled: users already labelled, for the diversity term (None = no diversity information yet)."""
    assign = fit_clusters(X_pool, n_clusters, seed)
    cl_pool = assign(X_pool)
    cl_lab = assign(X_labeled) if X_labeled is not None else np.array([], int)
    score = informativeness(model, X_pool, cl_pool, cl_lab, n_clusters, threshold, w)
    return np.argsort(-score)[:min(budget, len(X_pool))]


# ------------------------------------------------------------------ shared helpers
def _reservoir(seen, cap, rng):
    return seen if len(seen) <= cap else rng.choice(seen, cap, replace=False)


def _holdout(splits, gate_rows, seed, drift):
    """Random-holdout users from the validation split: gate users and early-stopping users (disjoint)."""
    Xva, tva, yva = splits["val"]
    perm = np.random.default_rng(seed).permutation(len(yva))
    g, e = perm[:min(gate_rows, len(yva) // 2)], perm[min(gate_rows, len(yva) // 2):]
    return (drift(Xva[g]), tva[g], yva[g]), (drift(Xva[e]), tva[e], yva[e])


def _evaluate(m, X, t, y):
    s = m.score(X)
    return s, qini(y, t, s), uplift_at(y, t, s, 0.10)


def _cost(m, X):
    times = []
    for _ in range(3):
        t0 = time.perf_counter(); m.score(X); times.append(time.perf_counter() - t0)
    return m.trees()["tau0"] + m.trees()["tau1"], 1e6 * min(times) / len(X)


def _stability(scores, idx):
    """Mean Spearman correlation and top-10% overlap between consecutive models' scores."""
    from scipy.stats import spearmanr
    k = len(idx) // 10
    top = lambda s: set(np.argsort(-s[idx])[:k])
    rho = [spearmanr(a[idx], b[idx])[0] for a, b in zip(scores, scores[1:])]
    ov = [len(top(a) & top(b)) / k for a, b in zip(scores, scores[1:])]
    return float(np.mean(rho)), float(np.mean(ov))


def _gate(cand, dep, gate, n):
    (lo, hi), _ = paired_boot(gate[2], gate[1], cand.score(gate[0]), dep.score(gate[0]), n=n)
    return lo, hi


# ------------------------------------------------------------------ active-learning simulation (Section 2.4)
def active_learning(splits, clf=None, reg=None, strategies=("random", "active (hybrid)"), drift_k=0.0,
                    cfg=None, seed=SEED, log=print):
    """Start from a model trained on init_frac of the training split; each cycle, choose batch_frac more users
    (at random or with the hybrid query strategy), reveal their outcomes, retrain on them plus the replay
    buffer, and gate the candidate against the deployed model on random-holdout users.
    drift_k > 0 shifts all raw features of the pool, holdout and test users by drift_k sd (labels where the
    model is out of date). Returns (per-cycle table, summary table)."""
    c = {**AL, **(cfg or {})}
    Xtr, ttr, ytr = splits["train"]; Xte, tte, yte = splits["test"]
    sd = Xtr[:, :len(RAW)].std(0)
    drift = (lambda X: shift(X, sd, drift_k)) if drift_k else (lambda X: X)
    Xp, Xt = drift(Xtr), drift(Xte)
    gate, es = _holdout(splits, c["gate_rows"], seed, drift)
    cl = clusters(Xp, c["n_clusters"], seed)
    stab_idx = np.random.default_rng(seed).choice(len(yte), min(c["stability_rows"], len(yte)), replace=False)
    fit = lambda idx: XLearner.fit(Xp[idx], ttr[idx], ytr[idx], *es, clf=clf, reg=reg)
    n, batch = len(ytr), int(c["batch_frac"] * len(ytr))
    rows, summ = [], []
    for strategy in strategies:
        rng = np.random.default_rng(seed)
        oracle = SimulatedOracle(ttr, ytr)
        seen = rng.choice(n, int(c["init_frac"] * n), replace=False); oracle.query(seen)
        dep = fit(seen); buf = _reservoir(seen, c["buffer_size"], rng)
        dep_scores, cand_scores, srows = [], [], []
        for cyc in range(c["rounds"] + 1):
            upd, lo, hi, promoted, cand = 0.0, np.nan, np.nan, None, dep
            if cyc > 0:
                free = np.ones(n, bool); free[seen] = False; pool = np.flatnonzero(free)
                new = rng.choice(pool, batch, replace=False) if strategy == "random" else \
                    pool[np.argsort(-informativeness(dep, Xp[pool], cl[pool], cl[seen], c["n_clusters"],
                                                     c["threshold"], (c["w_unc"], c["w_margin"], c["w_div"])))[:batch]]
                oracle.query(new)                                   # run the experiment on the chosen users
                t0 = time.perf_counter(); cand = fit(np.concatenate([new, buf])); upd = time.perf_counter() - t0
                lo, hi = _gate(cand, dep, gate, c["gate_boot"]); promoted = bool(lo > 0)
                if promoted:
                    dep = cand
                seen = np.concatenate([seen, new]); buf = _reservoir(seen, c["buffer_size"], rng)
            s, q, u10 = _evaluate(dep, Xt, tte, yte); dep_scores.append(s)
            cs, cq, cu = _evaluate(cand, Xt, tte, yte); cand_scores.append(cs)
            trees, us = _cost(dep, Xt)
            srows.append(dict(strategy=strategy, cycle=cyc, labeled_users=oracle.queried, candidate_qini=cq,
                              candidate_uplift10=cu, promoted=promoted, gate_ci_lo=lo, gate_ci_hi=hi,
                              deployed_qini=q, deployed_uplift10=u10, scoring_trees=trees,
                              scoring_us_per_user_batch=round(us, 3), update_s=round(upd, 1)))
            log(f"  {strategy:16s} cycle {cyc}: {oracle.queried:,} users | candidate Qini {cq:+.6f} | "
                f"deployed {q:+.6f} | gate [{lo:+.6f}, {hi:+.6f}]")
        rows += srows
        rho, ov = _stability(dep_scores, stab_idx); crho, cov = _stability(cand_scores, stab_idx)
        cands = [r["candidate_qini"] for r in srows[1:]]
        summ.append({"strategy": strategy, "final_candidate_qini": srows[-1]["candidate_qini"],
                     "mean_candidate_qini": float(np.mean(cands)), "final_deployed_qini": srows[-1]["deployed_qini"],
                     "promotions": f"{sum(r['promoted'] is True for r in srows)}/{c['rounds']}",
                     "deployed_top10_overlap": ov, "candidate_top10_overlap": cov,
                     "final_scoring_trees": srows[-1]["scoring_trees"],
                     "final_scoring_us_per_user_batch": srows[-1]["scoring_us_per_user_batch"],
                     "mean_update_s": float(np.mean([r["update_s"] for r in srows[1:]]))})
    return pd.DataFrame(rows), pd.DataFrame(summ).set_index("strategy").T


# ------------------------------------------------------------------ human label-curation simulation (Section 2.4)
def label_curation(splits, clf=None, reg=None, strategies=("random review", "targeted review"), cfg=None,
                   n_clusters=AL["n_clusters"], gate_rows=AL["gate_rows"], gate_boot=AL["gate_boot"],
                   seed=SEED, log=print):
    """Bot attack on one segment: fake conversions are added to treated users of one feature-space cluster in a
    random subset of the training split. Each cycle, annotators review a share of the remaining conversions
    (at random, or in the clusters with the most abnormal treated-control gap), flagged labels are corrected,
    a candidate is retrained on the corrected subset and gated against the deployed model.
    Returns (per-cycle table, summary table, info dict)."""
    c = {**CURATION, **(cfg or {})}
    Xtr, ttr, ytr = splits["train"]; Xte, tte, yte = splits["test"]
    gate, es = _holdout(splits, gate_rows, seed, lambda X: X)
    cl = clusters(Xtr, n_clusters, seed)
    stab_idx = np.random.default_rng(seed).choice(len(yte), min(AL["stability_rows"], len(yte)), replace=False)
    rng = np.random.default_rng(seed)
    sub = np.zeros(len(ytr), bool); sub[rng.choice(len(ytr), int(c["subset_frac"] * len(ytr)), replace=False)] = True
    sub_idx = np.flatnonzero(sub)
    bot = np.bincount(cl[sub & (ttr == 1)], minlength=n_clusters).argmax()
    cand_rows = np.flatnonzero(sub & (ttr == 1) & (ytr == 0) & (cl == bot))
    fake = rng.choice(cand_rows, min(len(cand_rows), int(c["fake_share"] * ytr[sub].sum())), replace=False)
    y_noisy = ytr.copy(); y_noisy[fake] = 1
    is_fake = np.zeros(len(ytr), bool); is_fake[fake] = True
    n_conv = int(y_noisy[sub].sum())
    fit = lambda y: XLearner.fit(Xtr[sub_idx], ttr[sub_idx], y[sub_idx], *es, clf=clf, reg=reg)
    q_clean = _evaluate(fit(ytr), Xte, tte, yte)[1]
    dep0 = fit(y_noisy)
    log(f"  {len(fake):,} fake conversions added to {int(ytr[sub].sum()):,} real ones (cluster {bot}); "
        f"clean-label Qini {q_clean:.6f}")

    def gap_z(y):
        """Per-cluster z-score of the treated-minus-control conversion rate (abnormal gaps first)."""
        z = np.zeros(n_clusters)
        for k in range(n_clusters):
            tm, cm = sub & (cl == k) & (ttr == 1), sub & (cl == k) & (ttr == 0)
            nt, nc = tm.sum(), cm.sum()
            if nt and nc:
                p = y[tm | cm].mean()
                z[k] = (y[tm].mean() - y[cm].mean()) / np.sqrt(max(p * (1 - p), 1e-12) * (1 / nt + 1 / nc))
        return z

    rows, summ = [], []
    for strategy in strategies:
        y, reviewed, dep = y_noisy.copy(), np.zeros(len(ytr), bool), dep0
        annot = SimulatedAnnotator(is_fake, c["hit_rate"], c["false_flag"], seed)
        budget, srows, scores = int(c["review_share"] * n_conv), [], []
        for cyc in range(c["cycles"] + 1):
            upd, lo, hi, promoted, cand = 0.0, np.nan, np.nan, None, dep
            if cyc > 0:
                conv = np.flatnonzero(sub & (y == 1) & (ttr == 1) & ~reviewed)
                pick = np.random.default_rng(seed + cyc).choice(conv, min(budget, len(conv)), replace=False) \
                    if strategy == "random review" else conv[np.argsort(-gap_z(y)[cl[conv]], kind="stable")[:budget]]
                reviewed[pick] = True
                y[annot.flag(pick, cyc)] = 0                    # corrected labels (also leave the replay buffer)
                t0 = time.perf_counter(); cand = fit(y); upd = time.perf_counter() - t0
                lo, hi = _gate(cand, dep, gate, gate_boot); promoted = bool(lo > 0)
                if promoted:
                    dep = cand
            s, q, u10 = _evaluate(dep, Xte, tte, yte); scores.append(s); cq = _evaluate(cand, Xte, tte, yte)[1]
            srows.append(dict(strategy=strategy, cycle=cyc, reviewed_conversions=int(reviewed.sum()),
                              fakes_left=int((is_fake & (y == 1)).sum()), candidate_qini=cq,
                              candidate_pct_of_clean=100 * cq / q_clean, promoted=promoted, gate_ci_lo=lo,
                              gate_ci_hi=hi, deployed_qini=q, deployed_pct_of_clean=100 * q / q_clean,
                              deployed_uplift10=u10, update_s=round(upd, 1)))
            log(f"  {strategy:16s} cycle {cyc}: reviewed {int(reviewed.sum()):,} | fakes left "
                f"{srows[-1]['fakes_left']:,} | candidate {srows[-1]['candidate_pct_of_clean']:.1f}% of clean")
        rows += srows
        gone = next((r["reviewed_conversions"] for r in srows if r["fakes_left"] <= 0.1 * len(fake)), None)
        rho, ov = _stability(scores, stab_idx)
        summ.append({"strategy": strategy, "final_candidate_pct_of_clean": srows[-1]["candidate_pct_of_clean"],
                     "final_deployed_pct_of_clean": srows[-1]["deployed_pct_of_clean"],
                     "fakes_removed_pct": 100 * (1 - srows[-1]["fakes_left"] / len(fake)),
                     "reviews_to_remove_90pct_fakes": gone if gone else "not reached",
                     "pct_of_conversions_reviewed_then": f"{100 * gone / n_conv:.0f}%" if gone else "-",
                     "promotions": f"{sum(r['promoted'] is True for r in srows)}/{c['cycles']}",
                     "deployed_rank_corr_between_cycles": rho, "deployed_top10_overlap_between_cycles": ov})
    info = dict(fakes=len(fake), real_conversions=int(ytr[sub].sum()), bot_cluster=int(bot), clean_qini=q_clean,
                noisy_qini=rows[0]["deployed_qini"])
    return pd.DataFrame(rows), pd.DataFrame(summ).set_index("strategy").T, info


# ------------------------------------------------------------------ plots
def plot_active_learning(df, path=None, title="Active learning with continual updates (retrain + replay)"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in df.groupby("strategy", sort=False):
        ax.plot(g.labeled_users, g.candidate_qini, marker="o", label=f"{name}: candidate")
        ax.plot(g.labeled_users, g.deployed_qini, ls="--", label=f"{name}: deployed")
    ax.set(xlabel="Experiment users labelled", ylabel="Test Qini", title=title)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_label_curation(df, info, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in df.groupby("strategy", sort=False):
        ax.plot(g.reviewed_conversions, g.candidate_qini, marker="o", label=f"{name}: candidate")
    ax.axhline(info["clean_qini"], c="k", ls="--", label="clean labels")
    ax.axhline(info["noisy_qini"], c="r", ls=":", label="before curation (deployed)")
    ax.set(xlabel="Conversions reviewed by humans", ylabel="Test Qini",
           title="Human label curation with continual updates (retrain + gate)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
