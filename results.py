import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self, predictions_path: str, ground_truth_path: str):
        self.predictions_path = predictions_path
        self.ground_truth_path = ground_truth_path
        self.y_pred = None
        self.y_true = None
        self.load_data()
    
    def load_data(self):
        """Loads predictions and ground truth data."""
        logger.info(f"Loading predictions from: {self.predictions_path}")
        self.y_pred = pd.read_csv(self.predictions_path).squeeze()
        logger.info(f"Predictions loaded: {self.y_pred.shape}")

        logger.info(f"Loading ground truth from: {self.ground_truth_path}")
        self.y_true = pd.read_pickle(self.ground_truth_path).squeeze()
        logger.info(f"Ground truth loaded: {self.y_true.shape}")
    
    def calculate_metrics(self):
        """Prints classification report and ROC-AUC score."""
        print("Classification Report:")
        classification_report1 = classification_report(self.y_true, self.y_pred, digits = 4, output_dict=True)
        #plot classification report

        # Convert to DataFrame (dropping support column to keep it clean)
        report_df = pd.DataFrame(classification_report1).T

        # Plot using Seaborn
        plt.figure(figsize=(8, 5))
        sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
        plt.title("Classification Report Heatmap")
        plt.xlabel("Metrics")

        # Save and show
        plt.show()
    
    def plot_confusion_matrix(self):
        """Displays the confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Clicked', 'Clicked'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
    
    def plot_precision_recall_curve(self):
        """Plots the precision-recall curve."""
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred)
        plt.plot(recall, precision, marker='.')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()

    def run_analysis(self):
        """Runs all analysis steps."""
        self.calculate_metrics()
        self.plot_confusion_matrix()
        self.plot_precision_recall_curve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model predictions.")
    parser.add_argument("--predictions", type=str, default ="data/predictions/predictions_valcatboost.csv" , help="Path to predictions CSV file")
    parser.add_argument("--ground-truth", type=str, default="data/processed/y_test.pkl", help="Path to ground truth pickle file")
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.predictions, args.ground_truth)
    analyzer.run_analysis()
