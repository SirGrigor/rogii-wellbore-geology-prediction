# Submission workflow — kernels-only Code competition

Confirmed via the Kaggle API: `is_kernels_submissions_only = True`, `max_daily_submissions = 5`,
metric labeled "Mean Squared Error" but **displayed as RMSE** (LB ~9.25 ↔ our 15.91 floor; MSE/RMSE
share the same optimum so modeling is unaffected). **`kaggle competitions submit` of a CSV returns
400 by design** — you submit by running a *notebook* that emits `/kaggle/working/submission.csv`.

## The split: train on Colab, infer on Kaggle

Inference notebooks usually run with **internet OFF**, so the Kaggle notebook cannot git-clone or
pip-install. Everything it needs is attached as **Datasets**:

```
Colab (GPU, heavy training)                 Kaggle (inference notebook, internet OFF)
─────────────────────────────              ──────────────────────────────────────────
src/train.py over GroupKFold-by-well   →   /kaggle/input/<comp>/        (auto-mounted data)
save model artifacts + probs/<v>/      →   /kaggle/input/rogii-models/  (your model dataset)
                                           /kaggle/input/rogii-code/    (this repo's src/)
                                           notebooks/kaggle_infer.ipynb → submission.csv → Submit
```

## One-time setup

1. **Code dataset** (`rogii-code`): upload this repo's `src/` folder as a Kaggle Dataset. The
   inference notebook does `sys.path.insert(0, "/kaggle/input/rogii-code")`. Re-upload (new version)
   whenever `src/` changes. (`src.config` degrades gracefully without kaggle-playground-utils, so the
   inference path needs only numpy/pandas/sklearn/lightgbm/dtaidistance — all preinstalled on Kaggle.)
2. **Model dataset** (`rogii-models`): on Colab, save your trained model(s) (e.g.
   `joblib.dump(model, "model.pkl")`) + `probs/<version>/`; upload as a Kaggle Dataset.

## Each submission

1. Train on Colab → refresh the `rogii-models` dataset (new version).
2. Open `notebooks/kaggle_infer.ipynb` on Kaggle, attach `rogii-code` + `rogii-models` + the
   competition data, **Save & Run All** (internet off).
3. Confirm it wrote `/kaggle/working/submission.csv` (14,151 rows), then **Submit to Competition**.
4. Record the LB score in the experiment diary (`python -m src.diary flag <version> "LB=..."`).

## Calibration note
Until a notebook submission succeeds, treat **GroupKFold OOF RMSE as the source of truth** and the
LB as a periodic calibration check (5/day budget). This matches the supervised-OOF discipline in
`docs/strategy.md` §2 — don't LB-probe.
