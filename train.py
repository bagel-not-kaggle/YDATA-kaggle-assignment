import argparse
import pandas as pd
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from features import create_features, get_target

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data():
    """
    Load preprocessed data and split into train/test
    """
    logger.info("Loading preprocessed data...")
    df = pd.read_csv('data/processed/cleaned_data.csv')

    # Create features using imported functions
    X = create_features(df)
    y = get_target(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    return (X_train, y_train), (X_test, y_test)


def train_model(model_name: str):
    """
    Train and evaluate model

    Args:
        model_name (str): Name of the model to train ('rf' for Random Forest)
    """
    # Get data
    logger.info(f"Training {model_name} model...")
    (X_train, y_train), (X_test, y_test) = get_data()

    # Initialize model based on model_name
    if model_name == 'rf':
        model = RandomForestClassifier(
            random_state=10,
            max_depth=30,
            min_samples_leaf=10,
            class_weight='balanced',
            n_jobs=-1
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'f1': f1_score(y_test, y_pred, average="weighted"),
        'precision': precision_score(y_test, y_pred, average="weighted"),
        'recall': recall_score(y_test, y_pred, average="weighted")
    }

    # Log metrics
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.3f}")

    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{model_name}_model.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CTR prediction model')
    parser.add_argument('-m', '--model-name', type=str, default='rf',
                        help='Model name (rf for Random Forest)')
    args = parser.parse_args()

    try:
        model, metrics = train_model(args.model_name)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise