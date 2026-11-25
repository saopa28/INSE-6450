"""Continual learning (Milestone 4, Section 1): replay buffer, drift-triggered updates (selected:
full retrain on new data + replay; alternatives: retrain on new data only, warm-start boosting + replay),
promotion gate, and the simulated drift experiment comparing the strategies."""
import gc
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .config import CONTINUAL, RAW, SEED, OUT_DIR
from .data import shift
from .metrics import qini, uplift_at, paired_boot, qini_floor, paired_boot_fast
from .model import XLearner
from .monitoring import Monitor, qini_check

# Drift intervals of the simulation: gradual covariate drift up to the shift that broke the
# Milestone 2 model in Milestone 3 (all raw features +3 sd), then a return to the original data.
SCHEDULE = [
    dict(name="Clean", k=0.0, cols=None),
    dict(name="f9 +0.5 sd", k=0.5, cols=["f9"]),
    dict(name="All raw +1 sd", k=1.0, cols=RAW),
    dict(name="All raw +2 sd", k=2.0, cols=RAW),
    dict(name="All raw +3 sd", k=3.0, cols=RAW),
    dict(name="All raw +3 sd (stable)", k=3.0, cols=RAW),
    dict(name="Clean (recurrence)", k=0.0, cols=None),
]

STRATEGIES = {
    "static": "No update (Milestone 2 model)",
    "retrain_new": "Full retrain on new batch",
    "retrain_replay": "Full retrain on new batch + replay (selected)",
    "warm_replay": "Warm-start + replay",
}


class ReplayBuffer:
    """Uniform reservoir sample over every labelled row seen so far. Uniform sampling keeps the
    true conversion rate (0.29%) and treated share (85%) that the X-learner's calibration relies on."""

    def __init__(self, capacity, n_features, seed=SEED):
        self.cap, self.seen, self.size = int(capacity), 0, 0
        self.X = np.empty((self.cap, n_features), np.float32)
        self.t = np.empty(self.cap, np.int8); self.y = np.empty(self.cap, np.int8)
        self.rng = np.random.default_rng(seed)

    def add(self, X, t, y):
        n = len(y)
        free = min(self.cap - self.size, n)
        if free:
            s = slice(self.size, self.size + free)
            self.X[s], self.t[s], self.y[s] = X[:free], t[:free], y[:free]
            self.size += free
        if n > free:   # Algorithm R: row i (1-based count) replaces a random slot with probability cap / i
            idx = self.seen + free + np.arange(1, n - free + 1)
            r = (self.rng.random(n - free) * idx).astype(np.int64)
            keep = r < self.cap
            self.X[r[keep]], self.t[r[keep]], self.y[r[keep]] = X[free:][keep], t[free:][keep], y[free:][keep]
        self.seen += n

    def get(self):
        return self.X[:self.size], self.t[:self.size], self.y[:self.size]


class PeakRSS:
    """Peak resident memory of the process during a block (XGBoost allocates outside Python)."""

    def __enter__(self):
        self.proc, self.peak, self.stop = psutil.Process(), 0, False
        self.base = self.proc.memory_info().rss
        self.th = threading.Thread(target=self._poll, daemon=True); self.th.start()
        return self

    def _poll(self):
        while not self.stop:
            self.peak = max(self.peak, self.proc.memory_info().rss); time.sleep(0.05)

    def __exit__(self, *a):
        self.stop = True; self.th.join()
        self.peak_mb = max(self.peak, self.proc.memory_info().rss) / 1e6
        self.delta_mb = (max(self.peak, self.proc.memory_info().rss) - self.base) / 1e6


def latency(model, X, batch, repeats=200):
    """Model-only scoring latency in ms (p50, p90) and throughput in rows/s, as in Milestone 3."""
    X = np.ascontiguousarray(X[: max(batch * 8, 4096)])
    for _ in range(10):
        model.score(X[:batch])
    lat = []
    gc.disable()
    try:
        for i in range(repeats):
            s = (i * batch) % (len(X) - batch + 1)
            t0 = time.perf_counter(); model.score(X[s:s + batch]); lat.append((time.perf_counter() - t0) * 1e3)
    finally:
        gc.enable()
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 90)), batch / (np.median(lat) / 1e3)


def _cat(*parts):
    return tuple(np.concatenate(z) for z in zip(*parts))


def make_update(strategy, deployed, new_tr, new_va, buffer, cfg):
    """Build a candidate model with the given strategy. Returns None for 'static'."""
    if strategy == "static":
        return None
    if strategy == "retrain_new":
        X, t, y = new_tr
        return XLearner.fit(X, t, y, *new_va, clf=deployed.clf, reg=deployed.reg)
    X, t, y = _cat(new_tr, buffer.get()) if buffer.size else new_tr
    if strategy == "retrain_replay":
        return XLearner.fit(X, t, y, *new_va, clf=deployed.clf, reg=deployed.reg)
    if strategy == "warm_replay":
        return XLearner.fit(X, t, y, *new_va, init=deployed, new_trees=cfg["new_trees"],
                            lr=cfg["warm_lr"], early_stop=cfg["warm_early_stop"])
    raise ValueError(strategy)


def run_simulation(v1, splits, strategies=tuple(STRATEGIES), schedule=SCHEDULE, cfg=None,
                   out_dir=OUT_DIR / "continual", seed=SEED, log=print, keep_models=False):
    """Simulated drift + continual update experiment. Each step: new labelled rows arrive (drawn from the
    training split, drifted), the deployed model of every strategy is checked (PSI + Qini trigger),
    triggered strategies build a candidate, the promotion gate decides, and every deployed model is
    evaluated on the untouched test split under the step's drift and under the original distribution.
    With keep_models=True also returns {step: {strategy: deployed model after the step}}."""
    cfg = {**CONTINUAL, **(cfg or {})}
    out_dir = Path(out_dir); (out_dir / "models").mkdir(parents=True, exist_ok=True)
    Xtr, ttr, ytr = splits["train"]; Xva, tva, yva = splits["val"]; Xte, tte, yte = splits["test"]
    sd = Xtr[:, :len(RAW)].std(0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ytr))
    rows_step = min(cfg["rows_per_step"], len(ytr) // len(schedule))
    a, b = int(cfg["split"][0] * rows_step), int((cfg["split"][0] + cfg["split"][1]) * rows_step)

    # Qini floor for one trigger window (half of a step's gate rows), from v1 on the clean validation split.
    window = (rows_step - b) // 2
    floor, fmean, fsd = qini_floor(yva, tva, v1.score(Xva), window, n=cfg["floor_boot"], seed=seed)
    if cfg.get("floor") is not None:      # manual override (e.g. for testing the update path)
        floor = float(cfg["floor"])
    log(f"rows per step {rows_step:,} | trigger window {window:,} rows | Qini floor {floor:.6f} "
        f"(mean {fmean:.6f}, sd {fsd:.6f})")

    buffer = ReplayBuffer(cfg["buffer_size"], Xtr.shape[1], seed)
    deployed = {s: v1 for s in strategies}
    version = {s: "v1" for s in strategies}
    monitors = {s: Monitor(Xtr) for s in strategies}
    cache, rows, updates, history = {}, [], [], {}

    def evaluate(model, vid, k_spec, Xtest):
        key = (vid, k_spec)
        if key not in cache:
            s = model.score(Xtest)
            cache[key] = (qini(yte, tte, s), uplift_at(yte, tte, s, 0.10), qini(yte, tte, s, auuc=True))
        return cache[key]

    test_orig = Xte
    for step, spec in enumerate(schedule):
        idx = order[step * rows_step:(step + 1) * rows_step]
        Xs = shift(Xtr[idx], sd, spec["k"], spec["cols"])
        ts, ys = ttr[idx], ytr[idx]
        new_tr, new_va, gate = (Xs[:a], ts[:a], ys[:a]), (Xs[a:b], ts[a:b], ys[a:b]), (Xs[b:], ts[b:], ys[b:])
        Xte_cur = shift(Xte, sd, spec["k"], spec["cols"])
        kspec = (spec["k"], tuple(spec["cols"] or ()))
        log(f"\nStep {step}: {spec['name']}")
        for s in strategies:
            dep, vid = deployed[s], version[s]
            mon = monitors[s].check(new_tr[0])
            triggered, wq = qini_check(dep, *gate, floor=floor, windows=2)
            q_before, u_before, a_before = evaluate(dep, vid, kspec, Xte_cur)
            row = dict(step=step, drift=spec["name"], strategy=s, version_before=vid,
                       psi_max=mon["psi_max"], psi_feature=mon["psi_feature"], monitor=mon["status"],
                       window_qini_1=wq[0], window_qini_2=wq[1], triggered=triggered,
                       qini_before=q_before, auuc_before=a_before, uplift10_before=u_before,
                       updated=False, promoted=False)
            if triggered and s != "static":
                gc.collect()
                with PeakRSS() as mem:
                    t0 = time.perf_counter()
                    cand = make_update(s, dep, new_tr, new_va, buffer, cfg)
                    upd_s = time.perf_counter() - t0
                (lo, hi), p_win = paired_boot(gate[2], gate[1], cand.score(gate[0]), dep.score(gate[0]),
                                              n=cfg["gate_boot"], seed=seed)
                promote = lo > 0
                row.update(updated=True, update_s=upd_s, peak_rss_mb=mem.peak_mb, rss_delta_mb=mem.delta_mb,
                           gate_ci_lo=lo, gate_ci_hi=hi, gate_p_better=p_win, promoted=promote,
                           cand_trees=sum(cand.trees().values()))
                log(f"  {s:15s} triggered | update {upd_s:6.1f}s | gate gain [{lo:+.6f}, {hi:+.6f}] -> "
                    f"{'PROMOTE' if promote else 'reject'}")
                if promote:
                    new_vid = f"{s}-step{step}"
                    cand.save(out_dir / "models" / new_vid, version=new_vid, parent=vid, step=step,
                              drift=spec["name"], strategy=s, update_seconds=round(upd_s, 2),
                              gate_qini_gain_ci=[lo, hi], buffer_rows=buffer.size,
                              training_rows=len(new_tr[2]) + (buffer.size if s != "retrain_new" else 0))
                    deployed[s], version[s] = cand, new_vid
                    monitors[s].set_reference(new_tr[0])
                    p50_1, p90_1, _ = latency(cand, Xte, 1)
                    p50_b, p90_b, thr = latency(cand, Xte, 256)
                    updates.append(dict(step=step, strategy=s, version=new_vid, parent=vid, update_s=upd_s,
                                        peak_rss_mb=mem.peak_mb, trees_scoring=cand.trees()["tau1"] + cand.trees()["tau0"],
                                        trees_total=sum(cand.trees().values()), scoring_size_mb=cand.size_mb(),
                                        p50_ms_b1=p50_1, p90_ms_b1=p90_1, p50_ms_b256=p50_b, p90_ms_b256=p90_b,
                                        throughput_rows_s=thr))
            dep, vid = deployed[s], version[s]
            row["version_after"] = vid
            row["qini_after"], row["uplift10_after"], row["auuc_after"] = evaluate(dep, vid, kspec, Xte_cur)
            row["qini_original"], row["uplift10_original"], row["auuc_original"] = evaluate(dep, vid, (0.0, ()), test_orig)
            rows.append(row)
            log(f"  {s:15s} {vid:22s} PSI {mon['psi_max']:7.3f} | trigger {str(triggered):5s} | "
                f"Qini current {row['qini_before']:+.6f} -> {row['qini_after']:+.6f} | original {row['qini_original']:+.6f}")
        if keep_models:
            history[step] = {s: (deployed[s], version[s]) for s in strategies}
        buffer.add(*new_tr)           # buffer holds past steps only when the next update is made
        del Xs, Xte_cur; gc.collect()

    res = pd.DataFrame(rows)
    upd = pd.DataFrame(updates)
    res.to_csv(out_dir / "continual_steps.csv", index=False)
    upd.to_csv(out_dir / "continual_updates.csv", index=False)
    (out_dir / "continual_config.json").write_text(json.dumps(
        dict(cfg=cfg, schedule=schedule, floor=floor, floor_mean=fmean, floor_sd=fsd, rows_per_step=rows_step), indent=2))
    return (res, upd, history) if keep_models else (res, upd)


def baseline_efficiency(model, X):
    p50_1, p90_1, _ = latency(model, X, 1)
    p50_b, p90_b, thr = latency(model, X, 256)
    return dict(version="v1", trees_scoring=model.trees()["tau1"] + model.trees()["tau0"],
                trees_total=sum(model.trees().values()), scoring_size_mb=model.size_mb(),
                p50_ms_b1=p50_1, p90_ms_b1=p90_1, p50_ms_b256=p50_b, p90_ms_b256=p90_b, throughput_rows_s=thr)


def plot_results(res, upd, out_dir=OUT_DIR / "continual"):
    """Metric trajectories across drift intervals, and update cost."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    names = res.groupby("step").drift.first()
    colors = dict(zip(STRATEGIES, ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    for s, g in res.groupby("strategy", sort=False):
        lab, c = STRATEGIES.get(s, s), colors.get(s)
        ax[0, 0].plot(g.step, g.qini_after, "o-", label=lab, color=c)
        ax[0, 1].plot(g.step, g.qini_original, "o-", label=lab, color=c)
        ax[1, 0].plot(g.step, g.uplift10_after * 100, "o-", label=lab, color=c)
        pr = g[g.promoted]
        ax[0, 0].scatter(pr.step, pr.qini_after, s=160, facecolors="none", edgecolors="k")
    ax[0, 0].set(title="Qini on test data under the current drift\n(circles: promoted update)", ylabel="Qini")
    ax[0, 1].set(title="Qini on the original test data (retention)", ylabel="Qini")
    ax[1, 0].set(title="Uplift@10% under the current drift", ylabel="uplift@10% (pp)")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.axhline(0, c="grey", lw=0.8); a.set_xticks(names.index); a.set_xticklabels(names.values, rotation=30, ha="right")
    ax[0, 0].legend(fontsize=8)
    if len(upd):
        w = 0.27
        for j, s in enumerate([s for s in STRATEGIES if s != "static"]):
            g = upd[upd.strategy == s]
            ax[1, 1].bar(g.step + (j - 1) * w, g.update_s, width=w, label=STRATEGIES[s], color=colors[s])
        ax[1, 1].set(title="Update time per promoted update", ylabel="seconds")
        ax[1, 1].set_xticks(names.index); ax[1, 1].set_xticklabels(names.values, rotation=30, ha="right")
        ax[1, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "continual_trajectories.png", dpi=150)
    return fig


METRICS = ["qini_after", "qini_original", "auuc_after", "uplift10_after"]


def run_seeds(v1, splits, seeds=(42, 1, 2, 3, 4), selected="retrain_replay",
              compare=("warm_replay", "retrain_new"), boot_steps=(3, 5, 6), boot_n=200,
              out_dir=OUT_DIR / "continual_seeds", cfg=None, log=print):
    """Repeat the simulation with several seeds (which users fall in each interval, the replay sample and the
    bootstrap resamples change; the test split and XGBoost settings do not). Then, for every seed, compare the
    selected strategy's deployed model with each alternative on the full test split with a paired bootstrap,
    at the steps in boot_steps (under that step's drift)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Xtr = splits["train"][0]; Xte, tte, yte = splits["test"]
    sd = Xtr[:, :len(RAW)].std(0)
    all_res, all_upd, boots = [], [], []
    for sd_i, seed in enumerate(seeds):
        log(f"\n================ seed {seed} ({sd_i + 1}/{len(seeds)}) ================")
        t0 = time.perf_counter()
        res, upd, hist = run_simulation(v1, splits, cfg=cfg, out_dir=out_dir / f"seed_{seed}", seed=seed,
                                        log=log, keep_models=True)
        all_res.append(res.assign(seed=seed)); all_upd.append(upd.assign(seed=seed))
        for step in boot_steps:
            spec = SCHEDULE[step]
            Xs = shift(Xte, sd, spec["k"], spec["cols"])
            a, va = hist[step][selected]
            s_a = a.score(Xs)
            for other in compare:
                b, vb = hist[step][other]
                if va == vb:
                    point, (lo, hi), p = 0.0, (0.0, 0.0), float("nan")
                else:
                    point, (lo, hi), p = paired_boot_fast(yte, tte, s_a, b.score(Xs), n=boot_n, seed=seed)
                boots.append(dict(seed=seed, step=step, drift=spec["name"], selected=va, other=other,
                                  other_version=vb, qini_gain=point, ci_lo=lo, ci_hi=hi, p_selected_better=p))
            del Xs
        del hist; gc.collect()
        log(f"seed {seed} done in {time.perf_counter() - t0:.0f} s")
    res, upd, boots = pd.concat(all_res, ignore_index=True), pd.concat(all_upd, ignore_index=True), pd.DataFrame(boots)
    summ = res.groupby(["step", "drift", "strategy"], sort=False).agg(
        **{f"{m}_mean": (m, "mean") for m in METRICS}, **{f"{m}_sd": (m, "std") for m in METRICS},
        seeds_triggered=("triggered", "sum"), seeds_promoted=("promoted", "sum")).reset_index()
    upd_summ = upd.groupby("strategy").agg(updates=("update_s", "size"), update_s_mean=("update_s", "mean"),
                                           update_s_sd=("update_s", "std"), peak_rss_mb_mean=("peak_rss_mb", "mean"),
                                           trees_scoring_mean=("trees_scoring", "mean"), size_mb_mean=("scoring_size_mb", "mean"),
                                           p50_ms_b1_mean=("p50_ms_b1", "mean"), p50_ms_b256_mean=("p50_ms_b256", "mean")).reset_index() if len(upd) else pd.DataFrame()
    for name, df in [("seeds_steps", res), ("seeds_summary", summ), ("seeds_updates", upd),
                     ("seeds_update_summary", upd_summ), ("seeds_bootstrap", boots)]:
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return res, summ, upd, upd_summ, boots


def plot_seeds(summ, out_dir=OUT_DIR / "continual_seeds"):
    """Mean +/- one standard deviation across seeds of Qini under the current drift and on the original data."""
    import matplotlib.pyplot as plt
    colors = dict(zip(STRATEGIES, ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]))
    names = summ.groupby("step").drift.first()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for s, g in summ.groupby("strategy", sort=False):
        for a, m in zip(ax, ["qini_after", "qini_original"]):
            mu, sdv = g[f"{m}_mean"].values, g[f"{m}_sd"].fillna(0).values
            a.plot(g.step, mu, "o-", color=colors.get(s), label=STRATEGIES.get(s, s))
            a.fill_between(g.step, mu - sdv, mu + sdv, color=colors.get(s), alpha=0.15)
    ax[0].set(title="Qini under the current drift (mean ± sd across seeds)", ylabel="Qini")
    ax[1].set(title="Qini on the original test data (retention, mean ± sd)", ylabel="Qini")
    for a in ax:
        a.axhline(0, c="grey", lw=0.8); a.set_xticks(names.index); a.set_xticklabels(names.values, rotation=30, ha="right")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "seeds_trajectories.png", dpi=150)
    return fig
