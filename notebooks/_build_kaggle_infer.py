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
# v6s4_fast (lean 4-model, stride-4) — SIMPLE-AVG (equal weights): sacred 9.166, which beat the
# OOF-optimized blend's 9.192 (the dev-tuned weights overfit). Equal weights generalize more safely
# to the unseen test wells. Sanity-check submission vs v5 (LB 9.644, sacred 9.155 — essentially tied).
WEIGHTS = {"v6s4_fast_lgb0": 0.25, "v6s4_fast_cat3": 0.25, "v6s4_fast_cat4": 0.25, "v6s4_fast_cat5": 0.25}


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": t.splitlines(keepends=True)}


cells = [
    md(f"""# ROGII — kernels-only submission (v5 ensemble, sacred 9.155)

Run **on Kaggle** (internet OFF) → writes `/kaggle/working/submission.csv` → **Submit to Competition**.
Attach: the competition data + **`rogii-code`** (repo `src/`) + **`rogii-models`** (probs/v5_lgb0, v5_cat3,
v5_cat4, v5_cat5 — each contains model_full.pkl). See docs/submission_workflow.md."""),
    md("## 1. Code on path + data location\n\n"
       "Auto-locates `src/` under /kaggle/input — works whether `rogii-code` is a manual `src/` upload "
       "OR a **GitHub-repo import** (which nests it under a `<repo>-main/` folder)."),
    code(f"""import sys, os, glob, zipfile, shutil
from pathlib import Path
import numpy as np

def _locate_src():
    # find kernel9251.py anywhere under /kaggle/input (flat files, a src/ folder, or...)
    h = glob.glob("/kaggle/input/**/kernel9251.py", recursive=True)
    if not h:                                              # ...inside a zip → extract first
        work = "/kaggle/working/_z"; shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            try: zipfile.ZipFile(z).extractall(work)
            except Exception: pass
        h = glob.glob(f"{{work}}/**/kernel9251.py", recursive=True)
    assert h, "rogii-code not found — attach the dataset with the src .py files"
    # re-package all the .py into a clean src/ package so `import src` works (any input layout)
    pkg = Path(h[0]).parent
    srcdir = Path("/kaggle/working/src"); shutil.rmtree(srcdir, ignore_errors=True); srcdir.mkdir(parents=True)
    for f in pkg.glob("*.py"): shutil.copy(f, srcdir / f.name)
    return Path("/kaggle/working")

root = _locate_src()
sys.path.insert(0, str(root)); print("src from:", root, "| files:", len(list((root/'src').glob('*.py'))))

# Auto-discover the competition data dir — robust to however Kaggle mounts it. We don't trust
# a hardcoded /kaggle/input/<comp> path; we find the dir that actually CONTAINS train/ + test/
# horizontal wells. (rogii-code/rogii-models have no such layout, so they won't false-match.)
def _find_data_dir():
    for d in sorted(glob.glob("/kaggle/input/*")):
        if glob.glob(f"{{d}}/**/test/*__horizontal_well.csv", recursive=True) \\
           and glob.glob(f"{{d}}/**/train/*__horizontal_well.csv", recursive=True):
            # RAW must be the dir whose children are train/ and test/
            t = glob.glob(f"{{d}}/**/test/*__horizontal_well.csv", recursive=True)[0]
            return str(Path(t).parent.parent)
    return None

_data = _find_data_dir()
if not _data:
    print("!! competition data NOT found. Full /kaggle/input tree:")
    for f in sorted(glob.glob("/kaggle/input/**", recursive=True))[:200]:
        print("  ", f)
    raise SystemExit("Attach the 'rogii-wellbore-geology-prediction' competition (Add Data → "
                     "Competitions) so train/+test/ mount under /kaggle/input.")
os.environ["ROGII_DATA_DIR"] = _data                       # so src.data reads the comp data
import joblib                                              # noqa: E402
from src import cv, data, kernel9251 as k9, submission     # noqa: E402
from src.config import TRAIN_DIR                            # noqa: E402
print("comp data:", _data, "| train:", len(data.list_well_ids("train")),
      "| test:", data.list_well_ids("test"))
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
# Discover models wherever the rogii-models dataset mounted (don't hardcode the path —
# Kaggle nests some inputs). Map {{model_name: path}} by the parent-dir name of each pkl.
_mp = {{Path(p).parent.name: p
       for p in glob.glob("/kaggle/input/**/model_full.pkl", recursive=True)}}
_mp.update({{Path(p).stem: p for p in glob.glob("/kaggle/input/**/v5_*.pkl", recursive=True)}})
print("models found:", sorted(_mp))
blend = np.zeros(len(Xt), dtype=float)
for name, w in WEIGHTS.items():
    assert name in _mp, f"model {{name}} not found (have {{sorted(_mp)}})"
    blend += w * joblib.load(_mp[name]).predict(Xt)
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
