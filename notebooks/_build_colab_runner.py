"""Generate notebooks/colab_runner.ipynb.

Ported from playground-s6e5 (2026-05-25) and adapted for rogii:
- Drive path → kaggle/rogii
- data flow uses `kaggle competitions download` (dirs of per-well CSVs, not flat files)
- dep auto-detect covers DTW / regression / neural-tabular, drops the F1-specific libs

Run once: `uv run python notebooks/_build_colab_runner.py`
Re-run after editing the cell sources below.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = "rogii-wellbore-geology-prediction"
COMP = "rogii-wellbore-geology-prediction"
GH = f"https://github.com/SirGrigor/{REPO}.git"  # NOTE: create this repo before using Colab
DRIVE = "/content/drive/MyDrive/Colab Notebooks/kaggle/rogii"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md(f"""# ROGII Wellbore — Colab runner

Generic notebook for running any `notebooks/*.py` training script on Colab Pro with GPU.

**Workflow** (run cells top to bottom):
1. Mount Drive + set Kaggle auth (from Colab Secrets)
2. Clone the repo (latest `main`)
3. **Version meta** — reads `SPRINT_ACTIVE.txt` to pick the active script
4. Install deps (auto-detected from the script's imports)
5. Get competition data (`kaggle competitions download`) + sync artifacts from Drive
6. Run the script
7. Sync artifacts back to Drive
8. Download the newest submission to local

To switch experiments: edit `SPRINT_ACTIVE.txt`, push, then re-run cells **clone → meta → install → run**.
"""),
    md("## 1. Mount Drive + Kaggle auth"),
    code(f"""from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '{DRIVE}'
# Note the space in 'Colab Notebooks' — Google's auto-created folder name; keep it.
!mkdir -p "$DRIVE_ROOT"
!ls -la "$DRIVE_ROOT" || echo 'Drive folder created (empty).'

# --- Kaggle CLI auth from Colab Secrets (key icon, left sidebar) ---
import os, json
try:
    from google.colab import userdata
    username = userdata.get('KAGGLE_USERNAME')
    api_token = userdata.get('KAGGLE_API_TOKEN')
    if api_token and api_token.strip().startswith('{{'):
        parsed = json.loads(api_token)
        api_token = parsed.get('key', api_token)
        username = username or parsed.get('username')
    if not username or not api_token:
        raise ValueError('missing username or token')
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = api_token
    print(f'✓ Kaggle auth set (user: {{username}})')
except Exception as exc:
    print(f'⚠ Kaggle auth skipped: {{exc}} — set KAGGLE_USERNAME + KAGGLE_API_TOKEN in Colab Secrets.')
"""),
    md("## 2. Clone repo + check out latest"),
    code(f"""%cd /content
!rm -rf {REPO}
!git clone -q {GH}
%cd {REPO}
!git log -1 --oneline
"""),
    md("## 3. Version meta — verify which script will run\n\n"
       "Edit `SPRINT_ACTIVE.txt` in the repo + push to change this. Re-run the clone cell first."),
    code(f"""import os
SCRIPT = None
EXTRA_ARGS = None
config_path = '/content/{REPO}/SPRINT_ACTIVE.txt'
if os.path.exists(config_path):
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(maxsplit=1)
                SCRIPT = parts[0]
                EXTRA_ARGS = parts[1] if len(parts) > 1 else ''
                break
    print(f'✓ Loaded from SPRINT_ACTIVE.txt:  {{SCRIPT}}  {{EXTRA_ARGS}}')
else:
    SCRIPT, EXTRA_ARGS = 'notebooks/01_eda.py', ''
    print(f'⚠ SPRINT_ACTIVE.txt not found, defaulted to: {{SCRIPT}}')

print('=' * 70)
print(f'ABOUT TO RUN:  {{SCRIPT}}   args: {{EXTRA_ARGS or "(none)"}}')
print('=' * 70)
!git log -1 --oneline
print(f'--- {{SCRIPT}} header ---')
!head -30 {{SCRIPT}}
"""),
    md("## 4. Install dependencies (auto-detected from the script's imports)\n\n"
       "We do NOT `pip install -e .` (its tool deps are editable local paths that don't exist on "
       "Colab). Instead: base deps + the shared **kaggle-playground-utils** from GitHub (public). "
       "`src.*` imports work because we run from the cloned repo root. **synth-decoder is local-only** "
       "— its GATE/adversarial checks are cheap and run on your machine, not on Colab GPU."),
    code("""import os
from pathlib import Path

script_text = Path(SCRIPT).read_text() if SCRIPT and Path(SCRIPT).exists() else ''
s = script_text.lower()

NEEDS_DTW      = 'dtaidistance' in s or 'src.align' in s or 'from src import align' in s
NEEDS_CATBOOST = 'catboost' in s
NEEDS_PYTABKIT = 'pytabkit' in s or 'realmlp' in s or 'tabm' in s
NEEDS_OPTUNA   = 'optuna' in s
NEEDS_TORCH    = 'torch' in s or NEEDS_PYTABKIT

!pip install -q uv

# Base deps + the shared toolkit from GitHub (public). NOT `-e .` — see the note above.
base = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'lightgbm', 'xgboost',
        'matplotlib', 'seaborn', 'dtaidistance']
if NEEDS_CATBOOST: base.append('catboost')
if NEEDS_PYTABKIT: base.append('pytabkit')
if NEEDS_OPTUNA:   base.append('optuna')
!uv pip install -q --system {' '.join(base)}
!uv pip install -q --system "git+https://github.com/SirGrigor/kaggle-playground-utils.git"

import lightgbm, sklearn, numpy, pandas
print(f'numpy {numpy.__version__}  pandas {pandas.__version__}  sklearn {sklearn.__version__}  lgb {lightgbm.__version__}')
import kaggle_playground_utils as _kpu; print(f'kaggle-playground-utils {_kpu.__version__}  (diary/observer/viz available)')
if NEEDS_TORCH:
    import torch
    print(f'torch {torch.__version__}  CUDA={torch.cuda.is_available()}  '
          f'device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
"""),
    md("## 5. Get competition data + sync prior artifacts from Drive"),
    code(f"""import os, shutil

for d in ('data/raw', 'data/external', 'data/splits', 'probs', 'submissions', 'reports'):
    os.makedirs(d, exist_ok=True)

# Competition data: download once per session (many small per-well CSVs → use the CLI).
if not os.listdir('data/raw') or not os.path.isdir('data/raw/train'):
    print('Downloading competition data...')
    !kaggle competitions download -c {COMP} -p data/raw
    !cd data/raw && (ls *.zip >/dev/null 2>&1 && unzip -o -q *.zip || echo 'no zip to unzip')
else:
    print('data/raw already populated.')

# Prior artifacts from Drive (probs / submissions / experiments diary)
for sub in ('probs', 'submissions'):
    src_root = f'{DRIVE}/{{sub}}'
    if os.path.isdir(src_root):
        for name in sorted(os.listdir(src_root)):
            s, d = f'{{src_root}}/{{name}}', f'{{sub}}/{{name}}'
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                for fn in os.listdir(s):
                    if not os.path.exists(f'{{d}}/{{fn}}'):
                        shutil.copyfile(f'{{s}}/{{fn}}', f'{{d}}/{{fn}}')
            elif not os.path.exists(d):
                shutil.copyfile(s, d)
        print(f'  synced {{sub}}/ from Drive')

print('--- data/raw ---')
!ls data/raw | head
"""),
    md("## 6. Run the target script"),
    code("""print(f'RUNNING: {SCRIPT}  {EXTRA_ARGS}')
import os; print('cwd:', os.getcwd())
!python {SCRIPT} {EXTRA_ARGS}
"""),
    md("## 7. Sync artifacts → Drive"),
    code(f"""import os, shutil
synced = False
for sub in ('probs', 'submissions'):
    if os.path.isdir(sub):
        for name in sorted(os.listdir(sub)):
            s = f'{{sub}}/{{name}}'
            d = f'{DRIVE}/{{sub}}/{{name}}'
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                for fn in os.listdir(s):
                    shutil.copyfile(f'{{s}}/{{fn}}', f'{{d}}/{{fn}}')
            else:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(s, d)
        print(f'  synced {{sub}}/ → Drive'); synced = True
if os.path.exists('experiments.jsonl'):
    shutil.copyfile('experiments.jsonl', f'{DRIVE}/experiments.jsonl')
    print('  synced experiments.jsonl → Drive'); synced = True
print('✓ sync complete.' if synced else 'ABORT: nothing to sync — did the run cell fail?')
"""),
    md("## 8. Download newest submission to local"),
    code(f"""from pathlib import Path
from google.colab import files
subs = sorted(Path('submissions').glob('*.csv'), key=lambda p: -p.stat().st_mtime)
if not subs:
    print('⚠ No submissions — the run cell probably failed.')
else:
    newest = subs[0]
    print(f'Downloading {{newest.name}}. Submit from local with:')
    print(f'  kaggle competitions submit -c {COMP} -f submissions/{{newest.name}} -m "from {{SCRIPT}}"')
    files.download(str(newest))
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent / "colab_runner.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}  ({len(cells)} cells)")
