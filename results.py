import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

def analyze_results(predictions_path: str, ground_truth_path: str):
    logger.info(f"Loading predictions from: {predictions_path}")
    y_pred = pd.read_csv(predictions_path)
    logger.info(f"Predictions loaded: {y_pred.shape}")

    logger.info(f"Loading ground truth from: {ground_truth_path}")
    y_true = pd.read_pickle(ground_truth_path)
    logger.info(f"Ground truth loaded: {y_true.shape}")


    # Calculate metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Clicked', 'Clicked'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model predictions.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV file")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth CSV file")
    args = parser.parse_args()

    analyze_results(args.predictions, args.ground_truth)
