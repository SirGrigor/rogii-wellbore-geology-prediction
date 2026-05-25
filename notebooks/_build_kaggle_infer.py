"""Generate notebooks/kaggle_infer.ipynb — the kernels-only submission notebook.

This competition is is_kernels_submissions_only=True: you submit by running a Kaggle
notebook that writes /kaggle/working/submission.csv. Such runs usually have INTERNET
OFF, so code + model artifacts must be attached as Kaggle **Datasets** (not git/pip).

Run once:  uv run python notebooks/_build_kaggle_infer.py
See docs/submission_workflow.md for the dataset handoff.
"""
from __future__ import annotations

import json
from pathlib import Path

COMP = "rogii-wellbore-geology-prediction"
CODE_DS = "rogii-code"      # Kaggle dataset holding this repo's src/ (rename to your slug)
MODEL_DS = "rogii-models"   # Kaggle dataset holding probs/<version>/ artifacts from Colab


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": t.splitlines(keepends=True)}


cells = [
    md(f"""# ROGII — Kaggle inference (kernels-only submission)

This competition accepts **notebook** submissions only. This notebook runs on Kaggle with
**internet OFF**, so it loads everything from attached **Datasets**:

1. **`{CODE_DS}`** — this repo's `src/` folder (upload it as a dataset).
2. **`{MODEL_DS}`** — trained artifacts `probs/<version>/{{oof,test}}.npy` (+ any saved models),
   produced on Colab and uploaded as a dataset.
3. The competition data is auto-mounted at `/kaggle/input/{COMP}/` (src.config detects it).

Output: `/kaggle/working/submission.csv`. Then **Submit** this notebook to the competition.
See `docs/submission_workflow.md` for the full handoff."""),
    md("## 1. Put attached code on the path + point at the data"),
    code(f"""import sys, os
# Attached code dataset (this repo's src/). Adjust the path to your dataset slug.
sys.path.insert(0, "/kaggle/input/{CODE_DS}")          # so `import src...` works
os.environ.setdefault("ROGII_DATA_DIR", "/kaggle/input/{COMP}")  # explicit; config also auto-detects

from src import data, features, submission           # noqa: E402
import numpy as np                                    # noqa: E402
print("test wells:", data.list_well_ids("test"))
"""),
    md("## 2. Build test features (post-PS rows = the submission rows)"),
    code("""ds = features.build_dataset("test", with_alignment=True, post_ps_only=True)
X_test, groups = ds["X"], ds["groups"]
print("X_test:", X_test.shape)
"""),
    md("## 3. Load model artifact(s) + predict\n\n"
       "Single model: load its saved booster and `.predict(X_test)`. Blend: load each member's "
       "test preds and combine with the OOF-fitted weights (see src.blend). Below is the "
       "single-model shape — adapt to your trained artifact."),
    code(f"""import joblib, glob
MODEL_DIR = "/kaggle/input/{MODEL_DS}"

# EXAMPLE (single LightGBM saved with joblib.dump(model, "model.pkl") on Colab):
model_paths = sorted(glob.glob(f"{{MODEL_DIR}}/**/*.pkl", recursive=True))
assert model_paths, f"no model artifacts found under {{MODEL_DIR}} — attach the {MODEL_DS} dataset"
model = joblib.load(model_paths[0])
test_pred = model.predict(X_test)

# (Blend variant: load several members' test preds + use src.blend.apply_blend with OOF weights.)
print("predictions:", test_pred.shape, "range", float(test_pred.min()), float(test_pred.max()))
"""),
    md("## 4. Assemble per-well predictions → submission.csv"),
    code("""# Map flat post-PS predictions back to per-well arrays (groups gives the well per row).
well_preds = {}
for wid in dict.fromkeys(groups):           # preserves order
    well_preds[wid] = test_pred[groups == wid]

ss = submission.build_submission_from_wells(well_preds, split="test")
ss.to_csv("/kaggle/working/submission.csv", index=False)
print("wrote /kaggle/working/submission.csv", ss.shape)
ss.head()
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent / "kaggle_infer.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}  ({len(cells)} cells)")
