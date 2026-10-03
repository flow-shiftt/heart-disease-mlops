"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.

Usage:
    python scripts/download_data.py           # skip download if raw file exists
    python scripts/download_data.py --force   # always re-download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiorisk import config  # noqa: E402
from cardiorisk.data import load_raw, missing_report, prepare, sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = prepare(force_download=args.force)
    raw = load_raw()
    print(f"raw file      : {config.RAW_FILE}  sha256={sha256(config.RAW_FILE)[:16]}")
    print(f"raw rows      : {len(raw)}")
    print(f"missing (raw) : {missing_report(raw).to_dict()}")
    print(f"clean file    : {config.CLEAN_FILE}  rows={len(df)}  cols={df.shape[1]}")
    print(f"positive rate : {df[config.TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
