"""Feature engineering and preprocessing.

Feature engineering lives *inside* the sklearn Pipeline (as a FunctionTransformer)
so the saved model accepts raw customer records.

Four candidate features were tested (see the ablation in the README / metrics.json).
Only ``log_tenure`` improved cross-validated ROC-AUC, so it is the only one used
by default; the others stay here so the ablation is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def _log_tenure(df: pd.DataFrame) -> pd.Series:
    # Churn risk falls steeply over the first months, then flattens out.
    return np.log1p(df["tenure"])


def _n_addons(df: pd.DataFrame) -> pd.Series:
    # How many optional services the customer pays for.
    return (df[ADDON_SERVICES] == "Yes").sum(axis=1)


def _auto_pay(df: pd.DataFrame) -> pd.Series:
    # Automatic payment removes a monthly moment of reconsidering.
    return df["PaymentMethod"].str.contains("automatic", case=False).astype(int)


def _charge_drift(df: pd.DataFrame) -> pd.Series:
    # Current bill minus historic average bill (price rise / upsell signal).
    avg_bill = np.where(df["tenure"] > 0,
                        df["TotalCharges"] / df["tenure"].clip(lower=1),
                        df["MonthlyCharges"])
    return df["MonthlyCharges"] - avg_bill


ENGINEERED_FEATURES = {
    "log_tenure": _log_tenure,
    "n_addons": _n_addons,
    "auto_pay": _auto_pay,
    "charge_drift": _charge_drift,
}
DEFAULT_EXTRAS = ("log_tenure",)


def add_features(df: pd.DataFrame, extras=DEFAULT_EXTRAS) -> pd.DataFrame:
    """Select the raw feature columns and append the requested engineered features."""
    out = df[config.FEATURE_COLS].copy()
    for name in extras:
        out[name] = ENGINEERED_FEATURES[name](out)
    return out


def build_preprocessor(scale_numeric: bool = True, extras=DEFAULT_EXTRAS) -> Pipeline:
    numeric = make_column_selector(dtype_include="number")
    categorical = make_column_selector(dtype_exclude="number")
    encode = ColumnTransformer(
        [
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features, kw_args={"extras": tuple(extras)})),
            ("encode", encode),
        ]
    )
