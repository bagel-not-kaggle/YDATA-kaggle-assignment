import argparse
import pandas as pd
import logging
from pathlib import Path
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from preprocess import Preprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, y_train, X_test, y_test, model_name: str, cat_features: list):
    """
    Train and evaluate a CatBoost model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name of the model to train ('catboost').
        cat_features (list): List of categorical feature indices or names.

    Returns:
        model (CatBoostClassifier): Trained model.
        metrics (dict): Evaluation metrics.
    """
    # Initialize the CatBoost model
    if model_name == 'catboost':
        model = CatBoostClassifier(random_seed=42, verbose=100, eval_metric='F1', cat_features=cat_features)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Train the model
    logger.info(f"Training {model_name} model...")
    X_train.drop(columns=['session_id', 'DateTime', 'user_id'], inplace=True)
    X_test.drop(columns=['session_id', 'DateTime', 'user_id'],  inplace=True)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Evaluate the model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = {
        'f1': f1_score(y_test, y_pred, average="weighted"),
        'precision': precision_score(y_test, y_pred, average="weighted"),
        'recall': recall_score(y_test, y_pred, average="weighted"),
    }

    # Log metrics
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.3f}")

    # Save the model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{model_name}_model.cbm'
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory with preprocessed data files")
    parser.add_argument("--model_name", type=str, default="catboost", help="Name of the model to train (default: catboost)")
    parser.add_argument("--cat_features", type=str, nargs='+', help="List of categorical feature names or indices")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Check if all required files are available
    """
    required_files = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(f"Required file '{file}' not found in {data_dir}")
    """
    # Load the preprocessed data
    logger.info(f"Loading preprocessed data from {data_dir}...")
    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    X_test = pd.read_pickle(data_dir / "X_test.pkl")
    y_train = pd.read_pickle(data_dir / "y_train.pkl").squeeze()    # Squeeze to Series for CatBoost
    y_test = pd.read_pickle(data_dir / "y_test.pkl").squeeze()     # Squeeze to Series for CatBoost
    # Determine categorical features
    if args.cat_features:
        # If categorical features are specified by names
        cat_features = [col for col in args.cat_features if col in X_train.columns]
    else:
        # Automatically detect categorical features based on their data types
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category").cat.add_categories("missing").fillna("missing")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category").cat.add_categories("missing").fillna("missing")


    logger.info(f"Categorical features: {cat_features}")

    # Train the model
    model, metrics = train_model(X_train, y_train, X_test, y_test, args.model_name, cat_features)
    logger.info("Training complete.")
