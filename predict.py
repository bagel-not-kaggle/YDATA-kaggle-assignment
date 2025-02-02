import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
from preprocess import Preprocessor


class Predictor:
    def __init__(self, model_path: Path, preprocessor_path: Path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        report = classification_report(y, predictions)
        print(report)
        return report

    def save_results(self, predictions, file_path: Path):
        np.savetxt(file_path, predictions, delimiter=',', fmt='%d')


if __name__ == '__main__':
    # --- Evaluation on Processed Data with Ground Truth ---
    test_data_path = Path("data/processed/cleaned_data.csv")
    data = Preprocessor.load_data(test_data_path)

    # Preserve all columns needed for feature engineering (e.g. DateTime)
    X = data.copy()
    if 'is_click' not in X.columns:
        raise ValueError("Test data must include 'is_click' column as ground truth")
    y = X.pop('is_click')

    # Load the saved model and preprocessor
    predictor = Predictor(
        model_path=Path("models/model.joblib"),
        preprocessor_path=Path("models/preprocessor.joblib")
    )

    # Transform the raw features using the saved preprocessor
    X_processed = predictor.preprocessor.transform(X)

    # Evaluate the model and print the classification report
    predictor.evaluate(X_processed, y)

    # --- Generate Predictions on the Real Test File (Without Ground Truth) ---
    real_test_path = Path("data/raw/X_test_1st.csv")
    real_data = Preprocessor.load_data(real_test_path)

    # Preserve all columns needed for feature engineering
    X_real = real_data.copy()  # Note: there's no is_click column here

    # Transform the real test data using the saved preprocessor
    X_real_processed = predictor.preprocessor.transform(X_real)

    # Generate predictions for the real test data
    real_predictions = predictor.predict(X_real_processed)

    # Save predictions to a CSV file
    output_file = Path("predictions_real.csv")
    predictor.save_results(real_predictions, output_file)
    print(f"Real test predictions saved to {output_file}")
