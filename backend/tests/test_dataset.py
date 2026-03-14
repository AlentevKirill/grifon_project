from __future__ import annotations

import pandas as pd
import pytest

from app.ml.dataset import detect_target_column, normalize_binary_target


def test_detect_target_column_case_insensitive() -> None:
    columns = ["TransactionID", "Amount", "Is_Fraud"]
    assert detect_target_column(columns) == "Is_Fraud"


def test_detect_target_column_raises_when_missing() -> None:
    with pytest.raises(ValueError):
        detect_target_column(["amount", "merchant"])


def test_normalize_binary_target_from_strings() -> None:
    target = pd.Series(["fraud", "legit", "true", "false", "1", "0"])
    normalized = normalize_binary_target(target)
    assert normalized.tolist() == [1, 0, 1, 0, 1, 0]


def test_normalize_binary_target_raises_for_invalid_values() -> None:
    target = pd.Series(["fraud", "unknown"])
    with pytest.raises(ValueError):
        normalize_binary_target(target)
