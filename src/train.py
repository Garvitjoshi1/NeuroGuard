import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from .utils import get_logger
from .data_loader import load_data
from .preprocess import build_preprocessing_pipeline

logger = get_logger(__name__)

def main():
    data_path = "data/raw/dataset.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    