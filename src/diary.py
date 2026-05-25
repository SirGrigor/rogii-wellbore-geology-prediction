"""Project shim: the experiment diary CLI, wired to this project.

    python -m src.diary timeline | compare A B | report vN | regressions | flag vN "note" | render
"""
from __future__ import annotations

import sys

from . import config  # noqa: F401 — side effect: configure() is called

from kaggle_playground_utils.diary import (  # noqa: F401, re-export
    compare,
    regressions,
    render_all,
    timeline,
)
from kaggle_playground_utils.diary import cli as _cli

if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
