from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.preprocessing import TabularFeaturePreprocessor


def test_preprocessor_bounds_output_dimension_with_svd() -> None:
    row_count = 120
    dataframe = pd.DataFrame(
        {
            "amount": np.random.uniform(1.0, 500.0, size=row_count),
            "merchant": [f"merchant_{index}" for index in range(row_count)],
            "city": [f"city_{index % 15}" for index in range(row_count)],
        }
    )

    preprocessor = TabularFeaturePreprocessor(max_svd_components=32)
    train_matrix = preprocessor.fit_transform(dataframe)
    inference_matrix = preprocessor.transform(dataframe.iloc[:10].copy())

    assert train_matrix.dtype == np.float32
    assert inference_matrix.dtype == np.float32
    assert train_matrix.shape[0] == row_count
    assert inference_matrix.shape[0] == 10
    assert train_matrix.shape[1] <= 32
    assert inference_matrix.shape[1] == train_matrix.shape[1]

    metadata = preprocessor.feature_metadata_
    assert metadata.uses_dimensionality_reduction is True
    assert metadata.output_feature_count == train_matrix.shape[1]


def test_preprocessor_handles_tiny_training_split_without_reducer() -> None:
    dataframe = pd.DataFrame(
        {
            "amount": [42.0],
            "merchant": ["single_merchant"],
        }
    )

    preprocessor = TabularFeaturePreprocessor(max_svd_components=8)
    transformed = preprocessor.fit_transform(dataframe)

    assert transformed.dtype == np.float32
    assert transformed.shape == (1, preprocessor.output_dim_)

    metadata = preprocessor.feature_metadata_
    assert metadata.uses_dimensionality_reduction is False
