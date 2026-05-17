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
    def __init__(self, column: str = "smoking_status"):
        self.column = column
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, x):
        X = X.copy()
        X[self.column] = X[self.column].replace("Unknown", np.nan)
        return X
    
    def get_column_groups():
        numerical = ["age", "avg_glucose_level", "bmi"]
        binary_numeric = ["hypertension", "heart_disease"]
        categorical = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
        return numerical, binary_numeric, categorical
    
    def build_preprocessing_pipeline() -> ImbPipeline:
        numerical_cols, binary_cols, categorical_cols = get_column_groups()
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler)
        ])
        
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numerical_cols),
                ("bin", "passthrough", binary_cols),
                ("cat", cat_pipe, categorical_cols)
            ],
            remainder="drop"
        )
        
        full_pipeline = ImbPipeline([
            ("clean_smoking", CleanSmokingStatus()),
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42))
        ])
        
        logger.info("Preprocessing pipeline built successfully.")
        return full_pipeline