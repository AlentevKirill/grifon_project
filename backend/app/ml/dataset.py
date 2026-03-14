from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


KNOWN_TARGET_COLUMNS: tuple[str, ...] = (
    "is_fraud",
    "fraud",
    "label",
    "target",
    "class",
    "isfraud",
    "is_fraudulent",
    "fraudulent",
)


@dataclass(slots=True)
class DatasetBundle:
    """Container for training features and target."""

    features: pd.DataFrame
    target: pd.Series
    target_column: str


def detect_target_column(columns: Iterable[str]) -> str:
    """Detect the target column using common fraud-label naming conventions."""

    normalized_map = {column.strip().lower(): column for column in columns}

    for candidate in KNOWN_TARGET_COLUMNS:
        if candidate in normalized_map:
            return normalized_map[candidate]

    raise ValueError(
        "Unable to detect target column. Please include one of the following columns: "
        f"{', '.join(KNOWN_TARGET_COLUMNS)}"
    )


def normalize_binary_target(target: pd.Series) -> pd.Series:
    """Convert target values to 0/1 integers with strict validation."""

    normalized = target.copy()

    if normalized.dtype == object:
        normalized = normalized.astype(str).str.strip().str.lower().map(
            {
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "fraud": 1,
                "legit": 0,
                "legitimate": 0,
            }
        )

    normalized = pd.to_numeric(normalized, errors="coerce")

    if normalized.isna().any():
        raise ValueError("Target column contains non-binary values that cannot be normalized.")

    unique_values = set(normalized.unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError("Target column must be binary with values 0 and 1.")

    return normalized.astype(int)


def load_training_dataset(csv_path: str) -> DatasetBundle:
    """Load the fraud dataset and split into features and target column."""

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError("The training dataset is empty.")

    target_column = detect_target_column(dataframe.columns)
    target = normalize_binary_target(dataframe[target_column])
    features = dataframe.drop(columns=[target_column])

    if features.empty:
        raise ValueError("Dataset must contain feature columns in addition to the target column.")

    return DatasetBundle(features=features, target=target, target_column=target_column)
