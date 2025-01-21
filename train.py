import argparse
import pandas as pd
import logging
from pathlib import Path
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from preprocess import Preprocess
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, y_train, model_name: str, cat_features: list, val_size: float = 0.2):
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
        model = CatBoostClassifier(random_seed=42, verbose=100, eval_metric='F1', cat_features=cat_features,class_weights=[1, 10])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Train the model
    logger.info(f"Training {model_name} model...")
    X_train.drop(columns=['session_id', 'DateTime', 'user_id'], inplace=True)
    # Split X_train into training and validation sets
    X_train_final, X_valid, y_train_final, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )

    # Train the model
    logger.info(f"Training {model_name} model with validation set...")
    

    model.fit(X_train_final, y_train_final, eval_set=(X_valid, y_valid), use_best_model=True)

    # Save the model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{model_name}_model.cbm'
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    return model


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
    y_train = pd.read_pickle(data_dir / "y_train.pkl").squeeze()    # Squeeze to Series for CatBoost
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



    logger.info(f"Categorical features: {cat_features}")

    # Train the model
    model = train_model(X_train, y_train, args.model_name, cat_features)
    logger.info("Training complete.")
