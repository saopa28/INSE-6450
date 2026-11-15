# Milestone 3: AI-Based Digital Advertising Optimization Using Uplift Modeling

This folder contains the Milestone 3 code deliverables for the INSE-6450 project: failure analysis, robustness, monitoring, and adaptation of the Milestone 2 X-learner.

## Contents

| File or folder | What it is | Report sections |
|---|---|---|
| `Monitoring_Adaptation.ipynb` | Simulates a production data stream, computes drift and data-quality checks, renders the monitoring dashboard, saves versioned models, and runs the two adaptation experiments (moderate and severe drift). | 3.3, 4.1, 4.2, 4.3 |
| `Milestone2_Feedback.ipynb` | Responds to the Milestone 2 feedback. Compares the X-learner with an S-learner, a T-learner, and a response model on the same split, over three seeds. Measures model-only latency, memory, operation counts, and campaign-level lift. | 4.5 |
| `xlearner_model/` | The Milestone 2 X-learner (version `v1`), saved so the notebooks can load it without retraining. | All |
| `v3/` | The X-learner retrained on severely drifted data in the adaptation experiment. | 4.2, 4.3 |

### `xlearner_model/`

| File | Contents |
|---|---|
| `mu1.json`, `mu0.json` | Stage 1 classifiers: conversion probability for treated and control users. |
| `tau1.json`, `tau0.json` | Stage 2 regressors: the two uplift estimates that are combined into the final score. |
| `metadata.json` | Selected tuning run, hyperparameters, treated share `g`, feature order, dataset sizes, and library versions. |

The uplift score is `g * tau0(x) + (1 - g) * tau1(x)`. Only the two stage 2 models are needed for scoring.

### `v3/`

| File | Contents |
|---|---|
| `mu1.ubj`, `mu0.ubj`, `tau1.ubj`, `tau0.ubj` | The four models of the retrained X-learner, in XGBoost binary format. |
| `manifest.json` | Version name, parent version, data snapshot ID, features, hyperparameters, a SHA-256 hash of each model file, and training time. |

`v3` was trained on data where all 12 raw features are shifted by three standard deviations. It should only be used on data with that shift.

## How to run

Open a notebook in Google Colab:

- Monitoring and adaptation: https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%203/Monitoring_Adaptation.ipynb
- Milestone 2 feedback: https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%203/Milestone2_Feedback.ipynb

Then run the cells in order from the top. No setup is needed:

- The dataset is downloaded from this repository's `v1.0-engineered` release.
- `Monitoring_Adaptation.ipynb` downloads the saved Milestone 2 model from `xlearner_model/`. If the download fails, it trains the model from scratch instead.

The saved model was written with XGBoost 3.4.1. If loading fails on an older version, run `!pip install xgboost==3.4.1` in a cell before the first cell.

## Runtime

Measured on a standard Colab CPU runtime.

| Notebook | Approximate time | Where the time goes |
|---|---|---|
| `Monitoring_Adaptation.ipynb` | 20 minutes | Two full retrains of the X-learner, about 6 minutes each. |
| `Milestone2_Feedback.ipynb` | 2.5 to 3 hours | Tuning four learners (about 35 minutes), twelve final fits over three seeds (about 90 minutes), and the paired bootstrap (about 30 minutes). |

## Existing outputs

The notebook cells have already been run, and the outputs are saved in the notebooks. The results, tables, and the monitoring dashboard can be reviewed without re-running anything.

## Reproducibility

- The train, validation, and test split is the same 70/15/15 stratified split as Milestone 2 (seed 42).
- The loaded Milestone 2 model reproduces the Milestone 2 test Qini of 0.000376.
- Files written by the notebooks (model versions, `monitoring_log.csv`, the dashboard image) are saved to `/content` in Colab and are lost when the session ends.
