# Milestone 4: Continual learning, human-in-the-loop, and the final pipeline

This folder holds the final code of the INSE 6450 uplift-modeling project. There are four deliverables:

| Deliverable | Files | What it covers |
|---|---|---|
| **1. Full pipeline** | **Code:** `uplift_pipeline/` (package) and `run_pipeline.py` (runs every stage)<br>**Run it in Colab:** `Run_Full_Pipeline.ipynb` (only calls `run_pipeline.py` and shows its results) | Data, model, robustness tests, monitoring, continual learning, human-in-the-loop and active learning, end to end |
| **2. Demo** | `Demo_End_to_End.ipynb` | One cycle of the deployed system: load → infer → detect drift → query human → update model |
| **3. Continual-learning simulation (Section 1.4)** | `Continual_Learning_Simulation.ipynb` | Seven drift intervals, four update strategies, five seeds. Part 1: main run (seed 42) with metrics before and after each update, efficiency, plots and saved versions. Part 2: mean ± sd across seeds, how often each strategy was promoted, paired bootstrap on the test split |
| **4. Active-learning and human-in-the-loop simulations (Section 2.4)** | `Active_learning_and_human_in_the_loop_simulation.ipynb` | Active learning (random vs hybrid query strategy: uncertainty, margin, diversity) and human label curation (random vs targeted review of bot conversions), both with retrain + replay updates and the promotion gate |

`requirements.txt` lists the Python dependencies.

### `uplift_pipeline/`

| Module | Stage |
|---|---|
| `config.py` | Seed, data URL, feature lists, Milestone 2 hyperparameters, Milestone 3 monitoring thresholds, Milestone 4 continual-learning settings. All paths are relative. |
| `data.py` | Loads the engineered Criteo Uplift v2 data, makes the Milestone 2 70/15/15 split (seed 42, stratified on treatment × conversion), and injects drift (shifts raw features and rebuilds the engineered ones). |
| `metrics.py` | Qini, AUUC, uplift@10%, paired bootstrap of the Qini gain, Qini floor. Same definitions as Milestones 2 and 3. |
| `model.py` | The X-learner: full training (used by the selected continual-learning update, retrain + replay), warm-start continuation (the alternative compared in Section 1.4), scoring, uncertainty, and versioned save/load with a manifest (parent version, hashes, configuration, metrics). `load_v1()` loads the Milestone 2 model from `../Milestone 3/xlearner_model/` or downloads it. |
| `robustness.py` | Milestone 3 stress tests (noise, rounding, masking, feature loss, +3 sd shift) with the pre-flight check. |
| `monitoring.py` | Data-quality and drift monitor (schema, nulls, out-of-range rows, PSI per feature) and the Qini retraining trigger. |
| `continual.py` | Replay buffer (reservoir sampling), update strategies, promotion gate, efficiency measurement (update time, peak memory, latency, size), the simulated drift experiment, and its plots. |
| `hitl.py` | Human-in-the-loop and active learning (Section 2): simulated reviewer for alerts and promotions, simulated annotator for label corrections, oracle (randomized experiment on the chosen users), the hybrid query strategy (uncertainty, margin, diversity), and the active-learning and label-curation simulations. |

## How to run

**Colab (recommended).** Open a notebook and choose Runtime > Run all. The first cell installs XGBoost 3.4.1, clones this repository if needed, and makes the package importable; the data is downloaded from the `v1.0-engineered` release.

- Continual learning (Section 1): https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%204/Continual_Learning_Simulation.ipynb
- Demo: https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%204/Demo_End_to_End.ipynb
- Active learning and human-in-the-loop (Section 2): https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%204/Active_learning_and_human_in_the_loop_simulation.ipynb
- Full pipeline: https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%204/Run_Full_Pipeline.ipynb

**Command line** (from a clone of the repository):

```bash
pip install -r "Milestone 4/requirements.txt"
cd "Milestone 4"
python run_pipeline.py                      # all stages, full data
python run_pipeline.py --frac 0.2           # quick check on 20% of the rows
python run_pipeline.py --stages continual   # one stage
python run_pipeline.py --train              # retrain v1 instead of loading the saved model
python run_pipeline.py --data path/to/criteo_uplift_engineered.zip
```

Approximate times on a Colab CPU runtime with the full data (14 million rows): continual-learning notebook 30-60 minutes (5 seeds), active-learning and human-in-the-loop notebook 30-40 minutes, demo 6-10 minutes, full pipeline about 60 minutes. Peak memory is about 5 GB.

## Outputs

Everything is written under `outputs/` in the working directory (in Colab, `/content/outputs/`, which is lost when the session ends; download what you need).

| Path | Contents |
|---|---|
| `outputs/continual/seed_<n>/continual_steps.csv` | One row per drift interval and strategy for that seed: PSI, monitor status, trigger, Qini/AUUC/uplift@10% before and after, retention, gate interval. |
| `outputs/continual/seed_<n>/continual_updates.csv` (and `efficiency.csv` for seed 42) | Per promoted update: update time, peak memory, trees, model size, latency (batch 1 and 256), throughput. |
| `outputs/continual/seed_42/continual_trajectories.png` | Figure 1.1: metric trajectories of the main run. |
| `outputs/continual/seed_<n>/models/<strategy>-step<k>/` | Every promoted model version: four boosters plus `manifest.json`. |
| `outputs/continual/seeds_summary.csv`, `seeds_bootstrap.csv`, `seeds_update_summary.csv`, `seeds_trajectories.png` | All seeds: mean ± sd per step and strategy, paired bootstrap results, update cost, and the mean ± sd plot. |
| `outputs/demo/` | Demo comparison table, review log, summary, and the promoted `v2-demo` model. |
| `outputs/hitl/` | Section 2.4 notebook: `active_learning.csv`, `active_learning_summary.csv`, `active_learning.png`, `label_curation.csv`, `label_curation_summary.csv`, `label_curation.png`. |
| `outputs/robustness.csv`, `monitoring.csv`, `active_learning*.csv`/`.png`, `label_curation*.csv`/`.png`, `pipeline_summary.json` | Results of `run_pipeline.py`. |

## Reproducibility

- Seed 42 everywhere; same split as Milestones 2 and 3. With the full data the loaded v1 model gives the Milestone 2 test Qini of 0.000376.
- No absolute paths: the package finds the Milestone 3 model relative to its own location and writes outputs relative to the working directory.
- The saved models were written with XGBoost 3.4.1; use that version (the notebooks install it).
- Model versions are saved with a manifest that records the parent version, the training rows, the gate result, the configuration, and SHA-256 hashes of the model files.

## Notes on the simulation

- The stream of new labelled users is drawn from the training split (disjoint rows per interval) and drifted; v1 saw those rows without drift. All reported metrics use the untouched test split.
- The oracle in the human-in-the-loop code is a randomized experiment on the users the model queries: in the simulation it reveals their logged treatment and conversion, which were randomized in the Criteo data.
