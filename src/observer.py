"""Project shim: re-export the shared experiment observer.

Importing this (or src.config) wires the diary to THIS project. Notebooks do::

    from src.observer import Experiment
"""
from __future__ import annotations

from . import config  # noqa: F401 — side effect: configure() is called

from kaggle_playground_utils.observer import (  # noqa: F401, re-export
    Experiment,
    MetricSpec,
    add_note,
    configure,
    jsonl_path,
)
