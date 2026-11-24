"""Shared settings for the uplift pipeline (Milestones 1-4).

All paths are relative to the working directory or derived from this file's
location, so the code runs unchanged in Colab or a local clone of the repo.
"""
from pathlib import Path

SEED = 42

# Data: the engineered Criteo Uplift v2 file published as a release of the repo.
DATA_URL = ("https://github.com/saopa28/INSE-6450/releases/download/"
            "v1.0-engineered/criteo_uplift_engineered.zip")

# Saved Milestone 2 X-learner (v1). Read from the local repo if present, otherwise downloaded.
REPO_ROOT = Path(__file__).resolve().parents[2]
V1_LOCAL_DIR = REPO_ROOT / "Milestone 3" / "xlearner_model"
V1_URL = "https://raw.githubusercontent.com/saopa28/INSE-6450/main/Milestone%203/xlearner_model"

# Outputs are written under the current working directory.
OUT_DIR = Path("outputs")

TARGET = "conversion"
RAW = [f"f{i}" for i in range(12)]
ENG = ["f4_x_f9", "f4_x_f11", "f9_x_f11", "f4_squared", "f9_squared", "f11_squared"]
FEATS = RAW + ENG
CRITICAL = ["f4", "f9", "f11"]

# Monitoring thresholds (Milestone 3, Sections 3 and 4.1).
PSI_WARN, PSI_BLOCK = 0.10, 0.25
NULL_MAX, NULL_MAX_CRITICAL, OOR_MAX = 0.01, 0.001, 0.01

# Continual learning (Milestone 4, Section 1).
CONTINUAL = dict(
    rows_per_step=1_000_000,   # new labelled rows arriving per drift interval
    split=(0.60, 0.15, 0.25),  # train / early-stopping validation / promotion-gate shares of each step
    buffer_size=200_000,       # replay buffer capacity (uniform reservoir over all past steps)
    new_trees=200,             # maximum trees added per warm-start update
    warm_lr=0.05,              # learning rate of the added trees (half the Milestone 2 rate)
    warm_early_stop=20,        # early-stopping patience for added trees
    gate_boot=100,             # paired bootstrap resamples for the promotion gate
    floor_boot=50,             # resamples used to set the Qini floor
)

# Milestone 2 configuration, used if metadata.json cannot be read.
CLF_DEFAULT = dict(objective="binary:logistic", n_estimators=1000, learning_rate=0.1, max_depth=6,
                   min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   tree_method="hist", eval_metric="logloss", early_stopping_rounds=30,
                   n_jobs=-1, random_state=SEED)
REG_DEFAULT = dict(objective="reg:squarederror", n_estimators=1000, learning_rate=0.1, max_depth=4,
                   min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                   tree_method="hist", early_stopping_rounds=30, n_jobs=-1, random_state=SEED)
