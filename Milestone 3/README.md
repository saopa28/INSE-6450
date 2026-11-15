# Milestone 3: AI-Based Digital Advertising Optimization Using Uplift Modeling

This folder contains the Milestone 3 deliverables for the INSE-6450 project. It includes the failure analysis, robustness/security assessment, monitoring and adaptation experiments, and model artifacts related to the Milestone 2 X-learner.

## Contents

| File or folder | What it is |
|---|---|
| `Anticipated_failure_risk_analysis_robustness_security.ipynb` | Notebook that analyzes anticipated failure modes, robustness to data drift and distribution shift, and security-related considerations for the uplift-modeling pipeline. |
| `Monitoring_Adaptation.ipynb` | Notebook that simulates a production data stream, runs drift and data-quality checks, renders the monitoring dashboard, saves versioned model artifacts, and executes the monitoring/adaptation experiments. |
| `Milestone2_Feedback.ipynb` | Notebook created in response to Milestone 2 feedback. It compares the X-learner against alternative uplift models (including S-learner, T-learner, and a response model) on the same split and reports results across multiple seeds. |
| `updated_comparison_of_xlearner_original_vs_updated.zip` | Compressed archive containing comparison outputs or supporting artifacts for the original vs. updated X-learner experiments. |
| `updated_xlearner_trained_model.zip` | Compressed archive containing the updated/retrained X-learner model artifact used in the later evaluation. |
| `xlearner_model/` | Saved Milestone 2 X-learner baseline model. This folder contains the trained stage-1 and stage-2 model components and their metadata. |
| `v3/` | Versioned retrained X-learner model produced for the adaptation experiment under severely drifted data. This is the updated model version designed for the shifted-feature scenario. |
| `README.md` | Overview of the folder contents and guidance for reproducing or reviewing the project deliverables. |

### `xlearner_model/`

| File | Contents |
|---|---|
| `mu1.json`, `mu0.json` | Stage 1 classifiers estimating the conversion probability for treated and control users. |
| `tau1.json`, `tau0.json` | Stage 2 regressors estimating uplift signals for the treated and control branches; these are combined into the final uplift score. |
| `metadata.json` | Saved metadata for the selected tuning run, including hyperparameters, treated share `g`, feature order, dataset sizes, and version information. |

The uplift score is calculated as:

`g * tau0(x) + (1 - g) * tau1(x)`

Only the two stage-2 models are required for scoring.

### `v3/`

| File | Contents |
|---|---|
| `mu1.ubj`, `mu0.ubj`, `tau1.ubj`, `tau0.ubj` | The four model files of the retrained X-learner in XGBoost binary format. |
| `manifest.json` | Version metadata including the version name, parent version, data snapshot ID, feature list, hyperparameters, hashes of each model file, and training time. |

`v3` was trained on data where all 12 raw features are shifted by three standard deviations. It should only be used on data with that same shift pattern.

## How to run

Open a notebook in Google Colab:

- Monitoring and adaptation: https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%203/Monitoring_Adaptation.ipynb
- Milestone 2 feedback: https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%203/Milestone2_Feedback.ipynb
- Failure-risk analysis/robustness/security: https://colab.research.google.com/github/saopa28/INSE-6450/blob/main/Milestone%203/Anticipated_failure_risk_analysis_robustness_security.ipynb

Then run the cells in order from the top. No additional setup is required:

- The dataset is downloaded from this repository's `v1.0-engineered` release.
- `Monitoring_Adaptation.ipynb` downloads the saved Milestone 2 model from `xlearner_model/`. If that download fails, it trains the model from scratch instead.

The saved model was written with XGBoost 3.4.1. If loading fails with an older version, run the following in a notebook cell before the first execution:

```python
!pip install xgboost==3.4.1
```

## Existing outputs

The notebook cells have already been run, and the outputs are saved in the notebooks. The results, tables, plots, and dashboard assets can be reviewed without re-running anything.

## Reproducibility

- The train, validation, and test split is the same 70/15/15 stratified split as Milestone 2 (seed 42).
- The loaded Milestone 2 model reproduces the Milestone 2 test Qini of 0.000376.
- Files written by the notebooks (model versions, `monitoring_log.csv`, and dashboard images) are saved to `/content` in Colab and are lost when the session ends.
