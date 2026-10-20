"""Downloading, loading and cleaning the Telco churn data.

Run ``python -m churn.data`` to download the dataset into ``data/raw/``.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

from churn import config


def download_data(
    url: str = config.DATA_URL,
    dest: Path = config.DATA_PATH,
    expected_sha256: str | None = config.DATA_SHA256,
    force: bool = False,
) -> Path:
    """Download the raw CSV to ``dest`` (skipped if it already exists)."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Data already present at {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)

    if expected_sha256:
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != expected_sha256:
            warnings.warn(
                f"Checksum mismatch for {dest.name}: got {digest}, expected "
                f"{expected_sha256}. Results may differ from the README.",
                stacklevel=2,
            )
    tmp.replace(dest)
    print(f"Saved to {dest}")
    return dest


def load_raw(path: Path = config.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m churn.data` to download it, or place "
            "Telco-Customer-Churn.csv there manually."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the schema and fix data-quality issues.

    * ``TotalCharges`` is stored as text and is blank for the 11 customers with
      ``tenure == 0`` (they haven't been billed yet), so blanks become 0.
    * ``Churn`` is mapped to 0/1 when present (it is absent when scoring new data).
    """
    missing = [c for c in config.REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    out = df.copy()
    for col in config.CATEGORICAL_COLS + [config.ID_COL]:
        out[col] = out[col].astype(str).str.strip()

    total = pd.to_numeric(out["TotalCharges"], errors="coerce")
    unbilled = total.isna() & (out["tenure"] == 0)
    if (total.isna() & ~unbilled).any():
        raise ValueError("TotalCharges has non-numeric values for customers with tenure > 0")
    out["TotalCharges"] = total.fillna(0.0)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)

    if config.TARGET in out.columns:
        labels = out[config.TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
        if labels.isna().any():
            raise ValueError("Churn column must contain only 'Yes' / 'No'")
        out[config.TARGET] = labels.astype(int)

    if out[config.ID_COL].duplicated().any():
        raise ValueError("Duplicate customerID values found")
    return out


def load_clean(path: Path = config.DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_data(force=force)
