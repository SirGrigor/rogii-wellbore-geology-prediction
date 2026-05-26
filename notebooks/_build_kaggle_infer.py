"""Generate notebooks/kaggle_infer.ipynb — the kernels-only SUBMISSION notebook (v5 ensemble).

Runs ON KAGGLE (internet OFF) to emit /kaggle/working/submission.csv, then Submit to Competition.
It re-computes inference: fits the spatial imputers on the dev wells (same deterministic split as
training), builds the 222 kernel9251 features for the 3 test wells (~20s), loads the 4 non-zero-weight
v5 models, applies the dev-OOF blend weights, de-residualizes (+ last_known_tvt), writes submission.csv.

Attach 2 Kaggle Datasets + the competition data (see docs/submission_workflow.md):
  - rogii-code   : this repo's `src/` folder (so `import src` works; needs only Kaggle-preinstalled
                   libs — numba/scipy/lightgbm/catboost/sklearn; src.config degrades w/o kaggle-playground-utils)
  - rogii-models : the trained models — upload probs/v5_lgb0, v5_cat3, v5_cat4, v5_cat5 (each has model_full.pkl)

Run once locally: uv run python notebooks/_build_kaggle_infer.py
"""
from __future__ import annotations

import json
from pathlib import Path

COMP = "rogii-wellbore-geology-prediction"
# v5 blend weights (dev-OOF optimized) — only the non-zero members (lgb1/lgb2 were zeroed).
WEIGHTS = {"v5_lgb0": 0.141, "v5_cat3": 0.114, "v5_cat4": 0.21, "v5_cat5": 0.535}


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": t.splitlines(keepends=True)}


cells = [
    md(f"""# ROGII — kernels-only submission (v5 ensemble, sacred 9.155)

Run **on Kaggle** (internet OFF) → writes `/kaggle/working/submission.csv` → **Submit to Competition**.
Attach: the competition data + **`rogii-code`** (repo `src/`) + **`rogii-models`** (probs/v5_lgb0, v5_cat3,
v5_cat4, v5_cat5 — each contains model_full.pkl). See docs/submission_workflow.md."""),
    md("## 1. Code on path + data location"),
    code(f"""import sys, os, glob, joblib
import numpy as np
sys.path.insert(0, "/kaggle/input/rogii-code")            # dataset root containing the `src` folder
os.environ["ROGII_DATA_DIR"] = "/kaggle/input/{COMP}"     # so src.data reads the comp data
from src import cv, data, kernel9251 as k9, submission     # noqa: E402
from src.config import TRAIN_DIR                            # noqa: E402
print("test wells:", data.list_well_ids("test"))
"""),
    md("## 2. Build the 222 features for the test wells (imputers fit on the same dev split)"),
    code("""dev, _ = cv.sacred_split(data.list_well_ids("train"))   # deterministic — matches training
k9.fit_imputers(dev, TRAIN_DIR)
test_df = k9.build_dataset([data.horizontal_path(w, "test") for w in data.list_well_ids("test")],
                           is_train=False, label="test_sub")
feats = [c for c in test_df.columns if c not in {"well", "id", "target"}]
Xt = test_df[feats].astype("float32")
anchor = test_df["last_known_tvt"].to_numpy(float)
print("test features:", Xt.shape)
"""),
    md("## 3. Load the v5 models, blend, de-residualize → submission.csv"),
    code(f"""WEIGHTS = {json.dumps(WEIGHTS)}
blend = np.zeros(len(Xt), dtype=float)
for name, w in WEIGHTS.items():
    hits = (glob.glob(f"/kaggle/input/rogii-models/**/{{name}}/model_full.pkl", recursive=True)
            or glob.glob(f"/kaggle/input/rogii-models/**/{{name}}.pkl", recursive=True))
    assert hits, f"model {{name}} not found under /kaggle/input/rogii-models"
    blend += w * joblib.load(hits[0]).predict(Xt)
tvt = blend + anchor                                        # de-residualize: drift + last-known TVT
ss = submission.build_submission(dict(zip(test_df["id"], tvt)))
ss.to_csv("/kaggle/working/submission.csv", index=False)
print("wrote submission.csv", ss.shape)
ss.head()
"""),
    md("**Now click _Submit to Competition_.** Our sacred-holdout estimate is ~9.16; the 3-well LB is "
       "a separate noisy check — record it in the diary but keep trusting sacred for decisions."),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent / "kaggle_infer.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}  ({len(cells)} cells)")
