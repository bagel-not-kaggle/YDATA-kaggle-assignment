import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from catboost import CatBoostClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(data: pd.DataFrame, model_path: str, batch_size: int = 30) -> np.ndarray:
    """
    Make predictions using the specified model.

    Args:
        data (pd.DataFrame): Input data for prediction.
        model_path (str): Path to the saved model.
        batch_size (int): Batch size for predictions.

    Returns:
        np.ndarray: Predictions from the model.
    """
    logger.info(f"Loading model from {model_path}...")
    model = CatBoostClassifier()
    model.load_model(model_path)  # Load the CatBoost model
    logger.info("Model loaded successfully.")

    # Placeholder for predictions (replace with actual prediction logic)
    #logger.info(f"Making predictions with batch size: {batch_size}")
    predictions = model.predict(data)  # Predict class labels
    #save predictions
    predict_df = pd.DataFrame(predictions, columns=['is_click'])
    predict_df.to_csv('predictions/predictions.csv', index=False)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--model-name", type=str, required=True, help="Path to the saved model file")
    parser.add_argument("--batch-size", type=int, default=32, help="Prediction batch size")
    args = parser.parse_args()

    # Load the data
    logger.info(f"Loading data from {args.data}...")
    data = pd.read_pickle(args.data)
    data.drop(columns=['session_id', 'DateTime', 'user_id'], inplace=True)
    cat_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        data[col] = data[col].astype("category").cat.add_categories("missing").fillna("missing")

    # Make predictions
    predictions = predict(data, args.model_name, args.batch_size)

    # Evaluate the model
    logger.info("Evaluating model...")
    y_test = pd.read_pickle("data/processed/y_test.pkl")
    y_pred = predictions > 0.5  # Apply thresholding if needed
    metrics = {
        'f1': f1_score(y_test, y_pred, average="weighted"),
        'precision': precision_score(y_test, y_pred, average="weighted"),
        'recall': recall_score(y_test, y_pred, average="weighted"),
    }

    # Log metrics
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.3f}")
