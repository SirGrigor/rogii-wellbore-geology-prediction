"""Signal-revelation report on rogii features (handover §7 move #2).

MEASURE which features carry the DRIFT signal and its SHAPE, BEFORE investing in the
alignment recipe — tells us what FE to build. Fit-free (MI + polyfit + binning via
synth_decoder.signal_revelation), so it runs locally (no model fit). Uses DEV wells
only; the sacred holdout stays untouched.

Run from repo root:  uv run python notebooks/02b_revelation.py [n_wells]
"""
from __future__ import annotations

import sys

from synth_decoder import signal_revelation as sr

from src import cv, data, features

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def main() -> None:
    dev, sacred = cv.sacred_split(data.list_well_ids("train"))
    sample = dev[:N]
    print(f"revelation on {len(sample)} dev wells (sacred {len(sacred)} untouched)")
    ds = features.build_dataset("train", well_ids=sample, with_alignment=True, target="residual")
    X, drift = ds["X"], ds["y"]            # y = TVT - anchor (the drift to model)
    print(f"X {X.shape} | drift: mean {drift.mean():.2f} std {drift.std():.2f} ft")
    sr.revelation_report(X, drift, task="regression", top_k=14)


if __name__ == "__main__":
    main()
