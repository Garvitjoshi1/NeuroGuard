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
from imblearn.pipeline import Pipeline as ImbPipeline
from .preprocess import build_full_pipeline

logger = get_logger(__name__)

def main():
    data_path = "data/raw/dataset.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    mlflow.set_experiment("stroke_prediction")
    logger.info("Experiment set: stroke_prediction")
    
    df = load_data(data_path)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    
    base_pipeline = build_preprocessing_pipeline()
    classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["roc_auc", "average_precision", "recall"]

    best_score = -1
    best_model_name = None
    best_pipeline = None
    
    for name, clf in classifiers.items():
        with mlflow.start_run(run_name=name):
            pipe = build_full_pipeline(clf)
            logger.info(f"Training {name} with 5‑fold CV")
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            mlflow.log_param("classifier", name)
            for sc in scoring:
                mean_val = np.mean(scores[f"test_{sc}"])
                std_val = np.std(scores[f"test_{sc}"])
                mlflow.log_metric(f"mean_{sc}", mean_val)
                mlflow.log_metric(f"std_{sc}", std_val)
                logger.info(f"{name} - {sc}: {mean_val:.4f} ± {std_val:.4f}")

            roc_auc_mean = np.mean(scores["test_roc_auc"])
            if roc_auc_mean > best_score:
                best_score = roc_auc_mean
                best_model_name = name
                # Clone and fit on all data
                from sklearn.base import clone
                best_pipeline = clone(pipe)
                best_pipeline.fit(X, y)

            mlflow.sklearn.log_model(pipe, "model")
            
    if best_pipeline is not None:
        model_path = model_dir / "best_model.pkl"
        joblib.dump(best_pipeline, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info(f"Best model: {best_model_name} with ROC AUC {best_score:.4f}, saved to {model_path}")
    else:
        logger.error("No model trained.")
        
if __name__ == "__main__":
    main()