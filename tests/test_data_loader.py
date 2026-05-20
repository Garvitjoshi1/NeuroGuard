import pytest
import pandas as pd
import os
from src.data_loader import load_data

def test_load_data_success(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2],
        "gender": ["Male", "Female"],
        "age": [45, 67],
        "hypertension": [0, 1],     
        "heart_disease": [0, 0],
        "ever_married": ["Yes", "No"],
        "work_type": ["Private", "Self-employed"],
        "Residence_type": ["Urban", "Rural"],
        "avg_glucose_level": [100, 150],
        "bmi": [25, 28],
        "smoking_status": ["never smoked", "smokes"], 
        "stroke": [0, 1]
    })
    
    fpath = tmp_path / "test.csv"
    df.to_csv(fpath, index=False)
    
    loaded = load_data(str(fpath))
    assert loaded.shape == (2, 11) 
    assert "id" not in loaded.columns
    assert loaded.columns[0] == "gender"
    
def test_load_data_missing_columns(tmp_path):
    df = pd.DataFrame({"x": [1]})
    fpath = tmp_path / "bad.csv"
    df.to_csv(fpath, index=False) 
    with pytest.raises(ValueError):
        load_data(str(fpath))