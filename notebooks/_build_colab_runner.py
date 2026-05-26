"""Generate notebooks/colab_runner.ipynb — a THIN, idempotent launcher.

Design: the notebook is a stable launcher that does ONLY the Colab-specific bits
(mount Drive, read the Kaggle secret into the env, fresh-clone the repo) and then calls
`colab/bootstrap.py`. ALL evolving run logic (install, data, run, Drive-sync) lives in
that versioned script, pulled fresh on every run — so re-running always executes the
latest correct flow and the notebook itself never needs editing.

To change the experiment: edit `SPRINT_ACTIVE.txt` + push, then re-run the Run cell.
Run once:  uv run python notebooks/_build_colab_runner.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = "rogii-wellbore-geology-prediction"
GH = f"https://github.com/SirGrigor/{REPO}.git"
DRIVE = "/content/drive/MyDrive/Colab Notebooks/kaggle/rogii"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": t.splitlines(keepends=True)}


cells = [
    md(f"""# ROGII — Colab runner (idempotent)

A **thin, stable launcher.** It only mounts Drive, sets Kaggle auth, and fresh-clones the
repo — then runs `colab/bootstrap.py`, which holds ALL the run logic (install, data, run,
Drive-sync) and is pulled fresh every time. **You never edit this notebook.**

- **Run an experiment:** Runtime → Run all.
- **Switch experiment / change the flow:** edit `SPRINT_ACTIVE.txt` (or `colab/bootstrap.py`)
  in the repo, push, then re-run the **Run** cell — it re-clones and picks up the latest.
- **Prereqs:** a Colab Secret `KAGGLE_API_TOKEN` (your 37-char token, notebook access ON).
  kaggle 2.x needs no username. Drive holds artifacts (probs/submissions/experiments.jsonl)."""),
    md("## 1. Setup — Drive + Kaggle auth (stable; rarely changes)"),
    code(f"""import os
from google.colab import drive
drive.mount('/content/drive')
os.environ['DRIVE_ROOT'] = '{DRIVE}'        # bootstrap syncs artifacts here
os.makedirs(os.environ['DRIVE_ROOT'], exist_ok=True)

# Secrets → env (🔑 icon, notebook access ON). bootstrap.py inherits these.
#   KAGGLE_API_TOKEN — required (kaggle 2.x bearer token, no username needed)
#   GH_TOKEN         — optional (fine-grained PAT, Contents:RW on this repo) to persist the diary to git
from google.colab import userdata
for name in ('KAGGLE_API_TOKEN', 'GH_TOKEN'):
    try:
        v = userdata.get(name)
        if v:
            os.environ[name] = v
            print(f'✓ {{name}} set')
        else:
            print(f'⚠ {{name}} secret is empty')
    except Exception:
        opt = ' (optional — diary won\\'t persist to git)' if name == 'GH_TOKEN' else ''
        print(f'⚠ no {{name}} secret{{opt}} — add via 🔑, notebook access ON')
"""),
    md("## 2. Run — fresh clone + bootstrap (all logic lives in the repo)"),
    code(f"""%cd /content
!rm -rf {REPO}
!git clone -q {GH}
%cd {REPO}
!git log -1 --oneline
# bootstrap.py inherits DRIVE_ROOT + KAGGLE_API_TOKEN from the env set above.
!python colab/bootstrap.py
"""),
    md("## 3. (optional) Download the newest submission to local\n\n"
       "Artifacts also sync to Drive automatically. For the kernels-only final submission you run "
       "`notebooks/kaggle_infer.ipynb` on Kaggle (see docs/submission_workflow.md) — not this."),
    code(f"""from pathlib import Path
from google.colab import files
subs = sorted(Path('/content/{REPO}/submissions').glob('*.csv'), key=lambda p: -p.stat().st_mtime)
if subs:
    print('newest:', subs[0].name); files.download(str(subs[0]))
else:
    print('no submissions yet')
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent / "colab_runner.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out}  ({len(cells)} cells)")
