"""Run the full Milestone 4 pipeline end to end and save every result under outputs/.

    python run_pipeline.py                         # everything, full data (about 45-60 min on Colab CPU)
    python run_pipeline.py --frac 0.1              # quick check on 10% of the rows
    python run_pipeline.py --stages model continual
    python run_pipeline.py --train                 # retrain v1 from scratch instead of loading it

Stages: model (load/train + test metrics), robustness, monitoring, continual (seed 42),
active_learning (active-learning and human label-curation simulations of Section 2).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uplift_pipeline import config as C                                   # noqa: E402
from uplift_pipeline.data import load_data, split, shift, raw_sd         # noqa: E402
from uplift_pipeline.metrics import summary                              # noqa: E402
from uplift_pipeline.model import XLearner, load_v1                      # noqa: E402
from uplift_pipeline.monitoring import Monitor                           # noqa: E402

STAGES = ["model", "robustness", "monitoring", "continual", "active_learning"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=C.DATA_URL, help="path or URL of criteo_uplift_engineered.zip")
    ap.add_argument("--frac", type=float, default=1.0, help="share of rows to use (default: all)")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--train", action="store_true", help="train v1 from scratch with the Milestone 2 configuration")
    ap.add_argument("--out", default=str(C.OUT_DIR))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(C.SEED)
    t0 = time.time()

    print(f"[data] loading {a.data} (frac={a.frac})")
    sp = split(load_data(a.data, frac=a.frac))
    (Xtr, ttr, ytr), (Xva, tva, yva), (Xte, tte, yte) = sp["train"], sp["val"], sp["test"]
    print(f"[data] train {len(ytr):,} | val {len(yva):,} | test {len(yte):,} rows")

    if a.train:
        print("[model] training v1 with the Milestone 2 configuration")
        v1 = XLearner.fit(Xtr, ttr, ytr, Xva, tva, yva)
        v1.save(out / "models" / "v1", version="v1", parent=None, note="trained by run_pipeline.py")
    else:
        v1 = load_v1()
    metrics = {"v1_test": summary(yte, tte, v1.score(Xte)), "v1_trees": v1.trees(), "rows": len(ytr) + len(yva) + len(yte)}
    print(f"[model] v1 test {metrics['v1_test']}")

    if "robustness" in a.stages:
        from uplift_pipeline.robustness import stress_tests
        df = stress_tests(v1, Xte, tte, yte, Monitor(Xtr))
        df.to_csv(out / "robustness.csv"); print(df.round(6).to_string())

    if "monitoring" in a.stages:
        mon, sd, rows = Monitor(Xtr), raw_sd(Xtr), []
        for name, k, cols in [("Clean", 0, None), ("f9 +0.5 sd", 0.5, ["f9"]), ("All raw +1 sd", 1, None),
                              ("All raw +3 sd", 3, None)]:
            Z = shift(Xte, sd, k, cols)
            r = mon.check(Z)
            rows.append(dict(batch=name, status=r["status"], psi_max=r["psi_max"], psi_feature=r["psi_feature"],
                             oor_rate=r["oor_rate"], alerts="; ".join(r["alerts"]), **summary(yte, tte, v1.score(Z))))
        df = pd.DataFrame(rows); df.to_csv(out / "monitoring.csv", index=False); print(df.round(6).to_string())

    if "continual" in a.stages:
        from uplift_pipeline.continual import run_simulation, plot_results, baseline_efficiency
        res, upd = run_simulation(v1, sp, out_dir=out / "continual")
        plot_results(res, upd, out / "continual")
        pd.DataFrame([baseline_efficiency(v1, Xte)]).to_csv(out / "continual" / "v1_efficiency.csv", index=False)

    if "active_learning" in a.stages:
        from uplift_pipeline.hitl import active_learning, label_curation, plot_active_learning, plot_label_curation
        print("[active learning] random vs hybrid query strategy, retrain + replay updates, promotion gate")
        al, al_summ = active_learning(sp, clf=v1.clf, reg=v1.reg)
        al.to_csv(out / "active_learning.csv", index=False); al_summ.to_csv(out / "active_learning_summary.csv")
        plot_active_learning(al, out / "active_learning.png")
        print("[human-in-the-loop] label curation: random vs targeted review of conversions")
        hc, hc_summ, info = label_curation(sp, clf=v1.clf, reg=v1.reg)
        hc.to_csv(out / "label_curation.csv", index=False); hc_summ.to_csv(out / "label_curation_summary.csv")
        plot_label_curation(hc, info, out / "label_curation.png")
        metrics["label_curation"] = info

    metrics["runtime_s"] = round(time.time() - t0, 1)
    (out / "pipeline_summary.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[done] {metrics['runtime_s']} s; outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
