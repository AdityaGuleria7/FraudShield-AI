"""Build model-input frames with consistent missing-value handling.

Training used pandas NaN for every missing value. scikit-learn's
SimpleImputer(missing_values=np.nan) treats NaN as missing in object columns
but NOT ``None``: an all-``None`` object column slips through as a literal
value, so a row's score used to depend on whether *other rows in the same
batch* happened to supply that column. Everything that feeds the model goes
through this helper so a row scores identically alone or in a batch.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def build_model_frame(
    records: Iterable[dict[str, Any]],
    features: list[str],
    categorical_features: Iterable[str],
) -> pd.DataFrame:
    categorical = [f for f in features if f in set(categorical_features)]
    cleaned = []
    for record in records:
        row = {}
        for feature in features:
            value = record.get(feature)
            missing = value is None or (isinstance(value, float) and np.isnan(value))
            row[feature] = np.nan if missing else value
        cleaned.append(row)
    frame = pd.DataFrame(cleaned, columns=features)
    if categorical:
        frame[categorical] = frame[categorical].astype(object)
    return frame
