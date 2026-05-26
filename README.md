# ROGII Wellbore Geology Prediction

Kaggle Featured competition — **tier-counting medal track** (deadline **2026-08-05**).

- **Task:** predict **True Vertical Thickness (TVT)** along horizontal wellbores from log
  curves (gamma-ray, resistivity, density) per depth, correlated to a vertical *type-well*
  reference. Tabular-**sequence** regression → geosteering automation.
- **Link:** https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- **Metric:** regression error (exact metric **confirmed in Phase-0 recon** — see
  `docs/reconnaissance.md`). Public-LB scores in starter notebooks were ~9.2–9.9.
- **Data:** per-well CSV pairs `{id}__horizontal_well.csv` + `{id}__typewell.csv`,
  train `{id}.png` cross-sections (reference viz, **not** the target), and a `.pptx` brief.

## Layout

```
src/                 comp-shaped code (config, data, cv, evaluate, features_sequence, align)
                     + thin shims re-exporting the shared experiment-diary tooling
notebooks/           EDA, baselines, experiments (+ colab_runner.ipynb)
docs/                reconnaissance.md, diary.md (auto-rendered), versions/<vN>.md
experiments.jsonl    append-only experiment diary — SOURCE OF TRUTH (git-tracked)
data/{raw,external,splits}/   downloaded + derived data (gitignored)
probs/ submissions/ reports/  artifacts (gitignored)
TOOLKIT_PORT.md      remaining S6E5 tools to port (blend_math, variant factory, radar)
```

## Tooling

Generic, battle-tested tools live in shared packages (depended on via local editable path),
not copy-pasted here — see `TOOLKIT_PORT.md` for the rationale and the port backlog:

- **`kaggle-playground-utils`** — experiment diary (`observer`/`diary`), model-comparison
  viz, signal-discovery probes (MI/adversarial, classification + regression),
  leakage-free encoders.
- **`synth-decoder`** — the GATE discipline + adversarial validation. (Leak/fingerprint
  modules assume *synthetic* data and **do not** apply here — rogii is real data.)

## Workflow

```bash
uv sync                              # install (pulls both shared toolkits editable)
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw
python -m src.diary timeline         # experiment diary
```

Every experiment goes through the diary discipline (hypothesis + predicted_delta BEFORE
training). **Cloud-first: all model fits run on Colab** — never locally.

### Colab (idempotent)
`notebooks/colab_runner.ipynb` is a thin, stable launcher: it mounts Drive, sets Kaggle
auth, fresh-clones the repo, and runs **`colab/bootstrap.py`** — which holds ALL run logic
(install · data · run the `SPRINT_ACTIVE.txt` script with `PYTHONPATH=repo root` · sync
artifacts to Drive). Because every run re-clones and bootstrap is versioned, **re-running
always executes the latest flow and the notebook never needs editing**. Switch experiments
by editing `SPRINT_ACTIVE.txt` (or `colab/bootstrap.py`) + push, then re-run the Run cell.
Sanity-check the flow locally with `python colab/bootstrap.py --dry-run`.
