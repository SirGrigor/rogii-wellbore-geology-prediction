"""Centralized configuration: seeds, paths, schema, metric.

Single source of truth — imported by every notebook and module so seeds, paths,
and the metric spec cannot drift. Importing this module also wires the shared
experiment-diary observer to this project (see the configure() call at the end).
"""
from __future__ import annotations

from pathlib import Path

from kaggle_playground_utils.observer import MetricSpec, configure

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
SPLITS = DATA / "splits"
PROBS = ROOT / "probs"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

# Raw data sub-layout (confirmed from `kaggle competitions files`):
#   raw/train/{id}.png                      cross-section viz (REFERENCE, not target)
#   raw/train/{id}__horizontal_well.csv     horizontal log curves (+ target columns)
#   raw/train/{id}__typewell.csv            vertical type-well log curves (reference)
#   raw/test/{id}__horizontal_well.csv      + raw/test/{id}__typewell.csv
#   raw/sample_submission.csv
#   raw/AI_wellbore_geology_prediction_task_en.pptx
TRAIN_DIR = RAW / "train"
TEST_DIR = RAW / "test"
SAMPLE_SUBMISSION = RAW / "sample_submission.csv"

# --------------------------------------------------------------------------- seeds
# Locked across the entire competition.
CV_SEED = 42
HOLDOUT_SEED = 11
MODEL_SEED = 7
N_FOLDS = 5
HOLDOUT_FRAC = 0.20

# --------------------------------------------------------------------------- problem
# Predict True Vertical Thickness (TVT, in feet) per 1-ft MD step along the horizontal
# well, BEYOND the Prediction Start (PS) point, by matching its GR signature to the
# type-well's GR-vs-TVT profile. CONFIRMED from the .pptx brief + data (Phase-0, 2026-05-25).
TARGET = "TVT"            # true target column in the TRAIN horizontal_well.csv
SUBMISSION_TARGET = "tvt"  # column name in sample_submission.csv (lowercase)
SUBMISSION_ID = "id"       # id format: "{well_id}_{row_index}" (rows AFTER the PS point)
WELL_ID = "well_id"        # derived from the filename ({id}__horizontal_well.csv)

# Metric — CONFIRMED (brief slide 14): RMSE of dTVT = (manualTVT - predictedTVT) over all
# predicted points. Feet units; LB leaders ~9.25. Thresholds are on the ~9-ft scale and
# are PROVISIONAL until the baseline establishes empirical fold variance (only 64 train wells).
METRIC = MetricSpec(
    name="rmse",
    greater_is_better=False,     # RMSE error → lower is better
    fold_collapse_drop=1.5,      # a fold > mean+1.5 ft = collapse (tune after baseline)
    leak_gap=0.5,                # |oof-holdout| ft gap above this looks like a leak
    regression_drop=0.1,         # holdout worse by >0.1 ft vs parent = regression
    fold_instability_std=1.0,    # ft std across folds above this = unstable
)

# --------------------------------------------------------------------------- schema (CONFIRMED Phase-0)
DEPTH_COL = "MD"                 # measured depth, 1-ft steps
TRAJECTORY_COLS = ["X", "Y", "Z"]  # 3D coordinates of each horizontal-well sample
CURVE_COLS = ["GR"]              # gamma-ray — the ONLY log curve; the alignment signal (may contain NaN)
TVT_INPUT_COL = "TVT_input"      # = true TVT until PS, naive carry-forward after PS. Strong anchor:
                                 # model the RESIDUAL (TVT - TVT_input) for scored (post-PS) rows.

# Usable horizontal-well features = columns present in BOTH train AND test:
FEATURE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]

# TRAIN-ONLY columns — geological formation top depths. NOT in test → DO NOT use as model
# features (feature-availability leakage). Use only for analysis / auxiliary targets.
TRAIN_ONLY_HORIZONTAL_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

# Type-well (vertical reference): TVT (depth axis) + GR. Train typewell ALSO has "Geology"
# (layer name) — train-only. Alignment target: map horizontal GR -> typewell (GR,TVT).
TYPEWELL_CURVE_COLS = ["GR"]
TYPEWELL_DEPTH_COL = "TVT"
TYPEWELL_TRAIN_ONLY_COLS = ["Geology"]

# --------------------------------------------------------------------------- wire diary
# One call: makes `from src.observer import Experiment` and `python -m src.diary`
# write to THIS project's experiments.jsonl with the regression-aware metric.
configure(root=ROOT, metric=METRIC, docs=DOCS)
