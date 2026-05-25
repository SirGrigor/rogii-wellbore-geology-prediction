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
# Predict True Vertical Thickness (TVT) per depth along the horizontal well.
TARGET = "TVT"          # PROVISIONAL — confirm exact column name from the data in Phase-0
WELL_ID = "well_id"     # derived from the filename ({id}__horizontal_well.csv)

# Metric — PROVISIONAL. Public-LB scores in starter notebooks were ~9.2-9.9, i.e.
# a regression ERROR (lower is better). Confirm the EXACT metric + direction from
# the competition Evaluation tab / the .pptx brief in Phase-0, then update this.
# Thresholds below are on the metric's own scale (~9 RMSE), NOT AUC's 0-1 scale.
METRIC = MetricSpec(
    name="rmse",                 # TODO Phase-0: confirm (RMSE? MAE? something custom?)
    greater_is_better=False,     # error metric → lower is better
    fold_collapse_drop=0.50,     # a fold > mean+0.5 error = collapse  (tune after EDA)
    leak_gap=0.30,               # |oof-holdout| error gap above this looks like a leak
    regression_drop=0.05,        # holdout error worse by >0.05 vs parent = regression
    fold_instability_std=0.40,   # error std across folds above this = unstable
)

# --------------------------------------------------------------------------- schema
# Log-curve columns — to be CONFIRMED from the CSV headers in Phase-0 EDA.
# Typical geosteering curves: gamma-ray (GR), resistivity (RES/RT), bulk density (RHOB).
# Depth references: measured depth (MD), true vertical depth (TVD).
DEPTH_COL = "MD"                 # TODO Phase-0: confirm measured-depth column name
CURVE_COLS: list[str] = []       # TODO Phase-0: fill from horizontal_well.csv header
TYPEWELL_CURVE_COLS: list[str] = []  # TODO Phase-0: fill from typewell.csv header

# --------------------------------------------------------------------------- wire diary
# One call: makes `from src.observer import Experiment` and `python -m src.diary`
# write to THIS project's experiments.jsonl with the regression-aware metric.
configure(root=ROOT, metric=METRIC, docs=DOCS)
