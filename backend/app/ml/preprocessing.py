from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


@dataclass(slots=True)
class FeatureSpaceMetadata:
    """Metadata describing preprocessed feature space dimensions."""

    original_feature_count: int
    transformed_feature_count: int
    output_feature_count: int
    uses_dimensionality_reduction: bool


class TabularFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Memory-safe preprocessing for mixed tabular fraud features.

    Design goals:
    - fit strictly on training data only;
    - keep high-cardinality one-hot outputs sparse;
    - bound dense feature dimension via TruncatedSVD before neural training.
    """

    def __init__(self, max_svd_components: int = 256) -> None:
        self.max_svd_components = max_svd_components

    @staticmethod
    def _build_column_transformer(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
        numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_columns = [column for column in features.columns if column not in numeric_columns]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                        dtype=np.float32,
                    ),
                ),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        return transformer, numeric_columns + categorical_columns

    @staticmethod
    def _to_dense_float32(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
        if sparse.issparse(matrix):
            return matrix.astype(np.float32).toarray()
        return np.asarray(matrix, dtype=np.float32)

    def _build_reducer(self, transformed_train: sparse.spmatrix | np.ndarray) -> TruncatedSVD | None:
        n_samples, n_features = transformed_train.shape
        max_allowed = min(self.max_svd_components, n_samples - 1, n_features - 1)

        # TruncatedSVD requires n_components >= 1 and < n_features.
        if max_allowed < 1:
            return None

        # Always compress feature spaces larger than the configured threshold.
        if n_features > self.max_svd_components or sparse.issparse(transformed_train):
            return TruncatedSVD(n_components=max_allowed, random_state=RANDOM_STATE)

        return None

    def fit(self, features: pd.DataFrame, y: pd.Series | None = None) -> "TabularFeaturePreprocessor":
        transformed_train = self.fit_transform(features, y=y)
        self.output_dim_ = int(transformed_train.shape[1])
        return self

    def fit_transform(self, features: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Preprocessor expects a pandas DataFrame as input.")

        transformer, ordered_features = self._build_column_transformer(features)
        transformed_train = transformer.fit_transform(features)

        reducer = self._build_reducer(transformed_train)
        if reducer is not None:
            reduced_train = reducer.fit_transform(transformed_train)
            output = np.asarray(reduced_train, dtype=np.float32)
        else:
            output = self._to_dense_float32(transformed_train)

        self.feature_columns_ = ordered_features
        self.column_transformer_ = transformer
        self.reducer_ = reducer
        self.feature_metadata_ = FeatureSpaceMetadata(
            original_feature_count=len(ordered_features),
            transformed_feature_count=int(transformed_train.shape[1]),
            output_feature_count=int(output.shape[1]),
            uses_dimensionality_reduction=reducer is not None,
        )
        self.output_dim_ = int(output.shape[1])

        return output

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "column_transformer_"):
            raise ValueError("Preprocessor is not fitted yet.")

        transformed = self.column_transformer_.transform(features)

        reducer: TruncatedSVD | None = getattr(self, "reducer_", None)
        if reducer is not None:
            reduced = reducer.transform(transformed)
            return np.asarray(reduced, dtype=np.float32)

        return self._to_dense_float32(transformed)
