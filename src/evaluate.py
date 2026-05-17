import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_predict
import joblib
from pathlib import Path
from .utils import get_logger
from .data_loader import load_data

logger = get_logger(__name__)

def plot_confusion_matrix(y_true, y_pred, path = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Prediction Label")
    plt.savefig(path, dpi = 100, bbox_inches = "tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {path}")
    
def plot_roc_curve(y_true, y_prob, path = "roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_curve = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color = "darkorange", lw = 2, label = f"ROC (area = {roc_curve:.2f})")
    plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Reciever Operating Characterstic")
    plt.legend(loc = "lower right")
    plt.savefig(path, dpi = 100, bbox_inches = "tight")
    plt.close()
    logger.info(f"ROC curve saved to {path}")
    
def evaluate():
    data_path = "data/raw/dataset.csv"
    model_path = "models/best_model.pkl"
    
    df = load_data(data_path)
    X = df.drop(columns = ["stroke"])
    y = df["stroke"]
    
    pipeline = joblib.load(model_path)
    
    y_pred = cross_val_predict(pipeline, X, y, cv=5, method="predict")
    y_prob = cross_val_predict(pipeline, X, y, cv=5, method="predict_proba")[:, 1]
    
    report = classification_report(y, y_pred, target_names=["No Stroke", "Stroke"])
    logger.info(f"\n{report}")
    
    plot_confusion_matrix(X, y)
    plot_roc_curve(y, y_prob)
    
if __name__ == "__main__":
    evaluate()