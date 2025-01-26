import argparse
import logging
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from prefect import task, flow
from sklearn.model_selection import train_test_split
import wandb

from preprocess_cat import DataPreprocessor
from train import ModelTrainer
from predict import predict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def preprocess_task(csv_path: str, output_path: str):
    logger.info("Starting the preprocessing task...")
    preprocessor = DataPreprocessor(output_path=Path(output_path))
    df = preprocessor.load_data(Path(csv_path))
    df_clean, X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test)

    # Log processed data as a WandB artifact
    artifact = wandb.Artifact("processed_data", type="dataset")
    artifact.add_dir(output_path)
    wandb.log_artifact(artifact)
    logger.info("Preprocessing task completed successfully.")

@task
def train_task(
    data_path: str,
    model_path: str,
    cat_features: list,
    val_size: float = 0.2,
    model_name: str = "catboost"
):
    logger.info("Starting the training task...")

    # Initialize wandb
    wandb.init(
        project="click-prediction",  # Replace with your project name
        entity="your-username",      # Replace with your W&B username or team
        group="training",            # Optional: Group related runs
        job_type="train",            # Optional: Tag this run as "train"
        config={
            "model_name": model_name,
            "val_size": val_size,
            "cat_features": cat_features,
        }
    )

    # Load data
    data_dir = Path(data_path)
    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    y_train = pd.read_pickle(data_dir / "y_train.pkl").squeeze()

    # Ensure categorical features are properly set
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = (
                X_train[col]
                .astype("category")
                .cat.add_categories("missing")
                .fillna("missing")
            )

    logger.info(f"Training {model_name} model...")

    # Set up the model
    if model_name == "catboost":
        model = CatBoostClassifier(
            random_seed=42,
            verbose=100,
            eval_metric="F1",
            cat_features=cat_features,
            class_weights=[1, 10],
            learning_rate=0.03,
            iterations=1000,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Split data into training and validation sets
    X_train_final, X_valid, y_train_final, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )

    # Train the model
    model.fit(X_train_final, y_train_final, eval_set=(X_valid, y_valid), use_best_model=True)

    # Log training metrics
    metrics = {
        "best_iteration": model.get_best_iteration(),
        "train_f1": model.best_score_["learn"]["F1"],
        "valid_f1": model.best_score_["validation"]["F1"],
    }
    wandb.log(metrics)

    # Save the trained model
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # Log the model as a WandB artifact
    artifact = wandb.Artifact("trained_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Finish the wandb run
    wandb.finish()

    logger.info("Training task completed successfully.")

@task
def predict_task(model_path: str, input_data_path: str, predictions_path: str):
    logger.info("Starting prediction...")
    predictions = predict(model_path=Path(model_path), input_data_path=Path(input_data_path))
    predictions.to_csv(predictions_path, index=False)

    artifact = wandb.Artifact("predictions", type="dataset")
    artifact.add_file(predictions_path)
    wandb.log_artifact(artifact)
    logger.info(f"Prediction completed. Results saved to {predictions_path}.")

@flow
def workflow(input_path: str, processed_path: str, model_path: str, predictions_path: str):
    logger.info("Starting the Prefect workflow...")
    try:
        run = wandb.init(
            project="click-prediction",
            entity="your-username",  # Replace with your W&B username or team
            group="example-group",
            job_type="flow"
        )
        preprocess_task(input_path, processed_path)
        train_task(processed_path, model_path, cat_features=[])
        predict_task(model_path, processed_path, predictions_path)
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Prefect workflow or individual tasks.")
    parser.add_argument("--input-path", type=str, help="Path to the raw input data.")
    parser.add_argument("--processed-path", type=str, help="Path to save the processed data.")
    parser.add_argument("--model-path", type=str, default="models/catboost_model.cbm", help="Path to save the trained model.")
    parser.add_argument("--predictions-path", type=str, help="Path to save the predictions.")
    parser.add_argument("--task", type=str, choices=["preprocess", "train", "predict"], help="Run a specific task.")
    args = parser.parse_args()

    if args.task == "preprocess":
        run = wandb.init(project="click-prediction", entity="your-username", group="example-group", job_type="preprocess")
        try:
            preprocess_task(args.input_path, args.processed_path)
        finally:
            wandb.finish()
    elif args.task == "train":
        run = wandb.init(project="click-prediction", entity="your-username", group="example-group", job_type="train")
        try:
            train_task(args.processed_path, args.model_path, cat_features=[])
        finally:
            wandb.finish()
    elif args.task == "predict":
        run = wandb.init(project="click-prediction", entity="your-username", group="example-group", job_type="prediction")
        try:
            predict_task(args.model_path, args.processed_path, args.predictions_path)
        finally:
            wandb.finish()
    else:
        workflow(
            input_path=args.input_path,
            processed_path=args.processed_path,
            model_path=args.model_path,
            predictions_path=args.predictions_path,
        )