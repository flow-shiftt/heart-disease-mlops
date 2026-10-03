"""Regenerate all EDA figures into reports/figures/ (headless; used in CI)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiorisk import config  # noqa: E402
from cardiorisk.data import load_clean, load_raw, prepare  # noqa: E402
from cardiorisk.eda import run_all  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not config.CLEAN_FILE.exists():
        prepare()
    for path in run_all(load_raw(), load_clean()):
        print(path.relative_to(config.PROJECT_ROOT))
