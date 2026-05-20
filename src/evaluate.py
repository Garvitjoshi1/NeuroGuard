# src/evaluate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.model_selection import cross_val_predict
import joblib
from pathlib import Path
from .utils import get_logger
from .data_loader import load_data

logger = get_logger(__name__)

def plot_confusion_matrix(y_true, y_pred, path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Stroke", "Stroke"],
                yticklabels=["No Stroke", "Stroke"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {path}")

def plot_roc_curve(y_true, y_prob, path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved to {path}")

def plot_precision_recall_curve(y_true, y_prob, path="precision_recall_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2,
             label=f"PR curve (AP = {avg_precision:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision‑Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Precision‑Recall curve saved to {path}")

def evaluate():
    data_path = "data/raw/dataset.csv"
    model_path = "models/best_model.pkl"

    df = load_data(data_path)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    pipeline = joblib.load(model_path)

    # Out-of-fold predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=5, method="predict")
    y_prob = cross_val_predict(pipeline, X, y, cv=5, method="predict_proba")[:, 1]

    # Classification report
    report = classification_report(y, y_pred, target_names=["No Stroke", "Stroke"])
    logger.info(f"\n{report}")

    # Additional metrics
    f1 = f1_score(y, y_pred)
    specificity = confusion_matrix(y, y_pred)[0,0] / (confusion_matrix(y, y_pred)[0,0] + confusion_matrix(y, y_pred)[0,1])
    logger.info(f"F1 Score (Stroke): {f1:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")

    # Generate plots
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_prob)
    plot_precision_recall_curve(y, y_prob)

if __name__ == "__main__":
    evaluate()