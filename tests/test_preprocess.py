import pandas as pd
import numpy as np
from src.preprocess import build_preprocessing_pipeline, CleanSmokingStatus

def test_clean_smoking_status():
    transformer = CleanSmokingStatus()
    df = pd.DataFrame({"smoking_status": ["Unknown", "never smoked", "smokes"]})
    transformed = transformer.transform(df)
    assert pd.isna(transformed.loc[0, "smoking_status"])
    assert transformed.loc[1, "smoking_status"] == "never smoked"

def test_preprocessing_pipeline():
    """Test that the preprocessing pipeline can fit and transform."""
    pipe = build_preprocessing_pipeline()
    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "age": [45, 67],
        "hypertension": [0, 1],
        "heart_disease": [0, 0],
        "ever_married": ["Yes", "No"],
        "work_type": ["Private", "Self-employed"],
        "residence_type": ["Urban", "Rural"],
        "avg_glucose_level": [100.0, 150.0],
        "bmi": [25.0, 28.0],
        "smoking_status": ["never smoked", "Unknown"],
    })
    # Fit on entire data (just for testing)
    pipe.fit(df)
    transformed = pipe.transform(df)
    # Should return a NumPy array with more columns (one-hot expansion)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 5   # expanded features