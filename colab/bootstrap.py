"""Colab bootstrap — the single source of truth for a Colab run (idempotent).

All evolving Colab logic lives HERE, version-controlled, so re-cloning the repo always
runs the latest correct flow. The notebook is a thin, stable launcher that only does the
Colab-specific bits (mount Drive, read the Kaggle secret into the env, clone/pull) and
then calls this. Change the run flow → edit this file + push; the notebook never changes.

Steps: install deps → get competition data (if missing) → pull prior artifacts from Drive
→ run the active script from SPRINT_ACTIVE.txt (with repo root on PYTHONPATH) → push
artifacts back to Drive.

Run on Colab from the repo root:  python colab/bootstrap.py
Env: DRIVE_ROOT (optional, for artifact sync), KAGGLE_API_TOKEN (for data download).
Use --dry-run to print the plan without installing/training (for local sanity).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rogii-wellbore-geology-prediction"
GH_REPO = "SirGrigor/rogii-wellbore-geology-prediction"
KPU_GIT = "git+https://github.com/SirGrigor/kaggle-playground-utils.git"
DEPS = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "lightgbm", "xgboost",
        "matplotlib", "seaborn", "dtaidistance", "joblib", "numba", "catboost", "rich"]
ARTIFACT_DIRS = ("probs", "submissions")

DRY = "--dry-run" in sys.argv


def sh(cmd: str) -> None:
    print(f"  $ {cmd}")
    if not DRY:
        subprocess.run(cmd, shell=True, check=True)


def active_script() -> str:
    for line in (ROOT / "SPRINT_ACTIVE.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise RuntimeError("SPRINT_ACTIVE.txt has no active script line")


def install() -> None:
    print("[1] deps (fixed set — idempotent)")
    sh("pip install -q -U kaggle")                      # Colab's preinstalled kaggle is old (1.x)
    sh(f"pip install -q {' '.join(DEPS)}")
    sh(f'pip install -q "{KPU_GIT}"')                    # shared toolkit (public); synth-decoder stays local


def get_data() -> None:
    print("[2] competition data")
    if (ROOT / "data" / "raw" / "train").is_dir():
        print("  data/raw/train present — skip download")
        return
    sh(f"kaggle competitions download -c {COMP} -p data/raw")
    sh("cd data/raw && unzip -o -q '*.zip' && rm -f *.zip")


def _sync(src: Path, dst: Path) -> None:
    if DRY:
        print(f"  sync {src} -> {dst}")
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def pull_artifacts() -> None:
    drive = os.environ.get("DRIVE_ROOT")
    print(f"[3] pull artifacts from Drive ({drive or 'DRIVE_ROOT unset — skip'})")
    if not drive:
        return
    for sub in ARTIFACT_DIRS:
        _sync(Path(drive) / sub, ROOT / sub)


def run_script() -> None:
    script = active_script()
    print(f"[4] RUN  {script}  (PYTHONPATH=repo root)")
    if DRY:
        print(f"  $ PYTHONPATH={ROOT} python {script}")
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(f"python {script}", shell=True, check=True, env=env, cwd=str(ROOT))


def push_artifacts() -> None:
    drive = os.environ.get("DRIVE_ROOT")
    print(f"[5] push artifacts to Drive ({drive or 'DRIVE_ROOT unset — skip'})")
    if not drive:
        return
    for sub in ARTIFACT_DIRS:
        _sync(ROOT / sub, Path(drive) / sub)
    src = ROOT / "experiments.jsonl"
    if src.exists() and not DRY:
        shutil.copy(src, Path(drive) / "experiments.jsonl")


def push_diary_to_git() -> None:
    """Persist the experiment diary to git (the source of truth) — needs GH_TOKEN.

    Colab runs append to the ephemeral clone's experiments.jsonl; without this the diary
    never reaches git. With a GH_TOKEN secret (fine-grained PAT, Contents:RW on this repo),
    render the diary + commit experiments.jsonl/docs and push to master. The token is never
    printed and never written to the on-disk remote config.
    """
    token = os.environ.get("GH_TOKEN")
    print(f"[6] persist diary to git ({'GH_TOKEN set' if token else 'no GH_TOKEN — diary stays on Drive only; skip'})")
    if not token or DRY:
        if DRY:
            print("  would: render diary, commit experiments.jsonl + docs/diary.md, push HEAD:master")
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-m", "src.diary", "render"], cwd=str(ROOT), env=env, check=False)
    for c in ('git config user.email "colab@rogii.bootstrap"',
              'git config user.name "rogii-colab-bootstrap"',
              "git add experiments.jsonl docs/diary.md docs/versions 2>/dev/null",
              'git diff --cached --quiet || git commit -q -m "diary: Colab run"',
              "git pull --rebase -q origin master"):
        subprocess.run(c, shell=True, cwd=str(ROOT), check=False)
    # push with the token inline (list args → not shell-echoed; URL kept out of logs)
    r = subprocess.run(["git", "push", "-q", f"https://{token}@github.com/{GH_REPO}.git", "HEAD:master"],
                       cwd=str(ROOT), capture_output=True, text=True)
    print("  ✓ diary pushed to git" if r.returncode == 0
          else f"  ⚠ diary push failed (rc={r.returncode}): {r.stderr.strip()[:200]}")


def gpu_verify() -> None:
    """S6E5-style LOUD GPU verification (run AFTER install, so torch is present).

    Prints torch CUDA + device name + nvidia-smi, exactly like S6E5's install cell, so the
    GPU status is unmistakable every run. If no GPU and ROGII_ALLOW_CPU != 1, ABORT here —
    fail fast (don't burn the 86-min FE then train on CPU). xgb device=cuda is the only
    working Colab-GPU GBDT path (lgb's pip wheel has no GPU build; that's why S6E5's visible
    GPU use was its torch NN models, not its GBDTs)."""
    print("=" * 64)
    has_proc = os.path.exists("/proc/driver/nvidia/version")
    # NOTE: do NOT import torch here — it pins a ~2-3GB CUDA context in this (parent) process
    # for the whole run, starving the training subprocess's RAM. nvidia-smi + /proc prove the GPU.
    if not DRY:
        subprocess.run("nvidia-smi -L 2>/dev/null || echo '[GPU] nvidia-smi: no GPU'", shell=True)
    print(f"[GPU] /proc/driver/nvidia={has_proc}  → GBDT path: "
          f"{'GPU available (xgb device=cuda)' if has_proc else 'CPU (no GPU attached)'}")
    print("=" * 64)
    if not DRY and not has_proc and os.environ.get("ROGII_ALLOW_CPU") != "1":
        raise SystemExit(
            "\n*** NO GPU ATTACHED to this Colab runtime. ***\n"
            "Runtime → Change runtime type → T4 GPU → Save (it reconnects), then re-run.\n"
            "Verify with `!nvidia-smi` first. To run on CPU anyway: set ROGII_ALLOW_CPU=1.\n")


def main() -> None:
    print(f"=== rogii Colab bootstrap (root={ROOT}{' DRY-RUN' if DRY else ''}) ===")
    install()
    gpu_verify()
    get_data()
    pull_artifacts()
    run_script()
    push_artifacts()
    push_diary_to_git()
    print("=== done ===")


if __name__ == "__main__":
    main()
