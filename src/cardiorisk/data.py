"""Data acquisition and cleaning for the UCI Cleveland heart-disease data."""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cardiorisk import config

log = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_raw(dest: Path = config.RAW_FILE, force: bool = False, timeout: int = 30) -> Path:
    """Fetch the raw Cleveland file. Tries the classic URL, then the zip bundle."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("Raw data already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(config.UCI_URL, timeout=timeout) as resp:
            payload = resp.read()
        log.info("Downloaded %d bytes from %s", len(payload), config.UCI_URL)
    except Exception as exc:  # noqa: BLE001 - any network failure triggers fallback
        log.warning("Primary download failed (%s); trying zip bundle", exc)
        with urllib.request.urlopen(config.UCI_ZIP_URL, timeout=timeout) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        payload = archive.read("processed.cleveland.data")

    dest.write_bytes(payload)
    log.info("Saved raw data to %s (sha256=%s)", dest, sha256(dest)[:12])
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the header-less raw file. Missing values are encoded as '?'."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with a binary target.

    Steps:
      * collapse diagnosis 1-4 into a single "disease present" class
      * cast discrete columns to nullable integers (keeps NaN for imputation)
      * blank out physiologically impossible values so the imputer handles them
      * drop exact duplicate rows
    Missing values are *not* filled here; imputation belongs inside the model
    pipeline so it is learned on training folds only (no leakage).
    """
    out = df.copy()
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])

    discrete = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + ["ca"]
    for col in discrete:
        out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")

    for col, (lo, hi) in config.VALID_RANGES.items():
        bad = (out[col] < lo) | (out[col] > hi)
        if bad.any():
            log.info("Nulling %d out-of-range values in %s", int(bad.sum()), col)
            out.loc[bad, col] = np.nan

    for col, levels in config.CATEGORY_LEVELS.items():
        bad = out[col].notna() & ~out[col].isin(levels)
        out.loc[bad, col] = pd.NA

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        log.info("Dropped %d duplicate rows", before - len(out))
    return out


def missing_report(df: pd.DataFrame) -> pd.Series:
    counts = df.isna().sum()
    return counts[counts > 0]


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + ["ca"]:
        df[col] = df[col].astype("Int64")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[config.INPUT_FEATURES].astype("float64")
    y = df[config.TARGET].astype(int)
    return X, y


def prepare(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean and persist the dataset. Returns the clean frame."""
    raw_path = download_raw(force=force_download)
    df = clean(load_raw(raw_path))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    log.info("Clean data: %d rows -> %s", len(df), config.CLEAN_FILE)
    return df
