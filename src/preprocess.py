import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from .utils import get_logger

logger = get_logger(__name__)


class CleanSmokingStatus(BaseEstimator, TransformerMixin):
    """Replace 'Unknown' with NaN in smoking_status column."""

    def __init__(self, column: str = "smoking_status"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].replace("Unknown", np.nan)
        return X


class ImputeAndScale(BaseEstimator, TransformerMixin):
    """Impute numerical columns with median, then scale."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.imputer.fit(X)
        X_imputed = self.imputer.transform(X)
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        X = self.imputer.transform(X)
        return self.scaler.transform(X)


def get_column_groups():
    """Return lists of column names for each type."""
    numerical = ["age", "avg_glucose_level", "bmi"]
    binary_numeric = ["hypertension", "heart_disease"]
    categorical = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
    return numerical, binary_numeric, categorical


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build a preprocessing-only pipeline (imputation + encoding + scaling).
    No SMOTE – output can be transformed. Used for deployment/inference.
    """
    numerical_cols, binary_cols, categorical_cols = get_column_groups()

    num_transformer = ImputeAndScale()

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("bin", "passthrough", binary_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        n_jobs=1,
    )

    pipeline = Pipeline([
        ("clean_smoking", CleanSmokingStatus()),
        ("preprocessor", preprocessor),
    ])

    logger.info("Preprocessing pipeline built successfully.")
    return pipeline


def build_full_pipeline(classifier) -> ImbPipeline:
    """
    Build a full training pipeline:
    clean_smoking → ColumnTransformer → SMOTE → classifier.
    This avoids nesting a scikit-learn Pipeline inside ImbPipeline.
    """
    numerical_cols, binary_cols, categorical_cols = get_column_groups()

    num_transformer = ImputeAndScale()

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("bin", "passthrough", binary_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        n_jobs=1,
    )

    # ImbPipeline with flat steps (no sklearn Pipeline as intermediate)
    full_pipe = ImbPipeline([
        ("clean_smoking", CleanSmokingStatus()),
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier),
    ])

    logger.info(f"Full pipeline built with classifier: {classifier.__class__.__name__}")
    return full_pipe