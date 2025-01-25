from prefect import task, flow
import pandas as pd
from pathlib import Path
from preprocess import DataPreprocessor
from train import ModelTrainer
from predict import predict  # Assuming predict is the main function in predict.py
import logging
from catboost import CatBoostClassifier
#from prefect.schedules import IntervalSchedule
from datetime import timedelta
from sklearn.model_selection import train_test_split
import wandb
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def preprocess_task(
    csv_path: str,
    output_path: str,
    wandb_group_id: str,
    remove_outliers: bool = False,
    fillna: bool = False,
    use_dummies: bool = True,
    save_as_pickle: bool = True
):
    """
    Task to preprocess data using the DataPreprocessor class and log artifacts to WandB.
    """
    logger.info("Starting the preprocessing task...")

    # Initialize WandB run with the shared group ID and "preprocess" job type
    with wandb.init(
        project="click-prediction",
        entity="wandb",
        group=wandb_group_id,
        job_type="preprocess"
    ) as run:
        preprocessor = DataPreprocessor(
            output_path=Path(output_path),
            remove_outliers=remove_outliers,
            fillna=fillna,
            use_dummies=use_dummies,
            save_as_pickle=save_as_pickle
        )

        # Load the raw data
        df = preprocessor.load_data(Path(csv_path))

        # Perform preprocessing
        logger.info("Running the preprocessing pipeline...")
        df_clean, X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

        # Save preprocessed data
        logger.info("Saving preprocessed data...")
        preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test)

        # Log processed data as a WandB artifact
        artifact = wandb.Artifact("processed_data", type="dataset")
        artifact.add_dir(output_path)
        run.log_artifact(artifact)

        logger.info("Preprocessing task completed successfully.")

@task
def train_task(
    data_path: str,
    model_path: str,
    cat_features: list,
    wandb_group_id: str,
    val_size: float = 0.2,
    model_name: str = "catboost"
):
    """
    Task to train a model, log metrics and parameters to WandB, and save the trained model.
    """
    logger.info("Starting the training task...")

    # Initialize WandB run with the shared group ID and "train" job type
    with wandb.init(
        project="click-prediction",
        entity="wandb",
        group=wandb_group_id,
        job_type="train"
    ) as run:
        logger.info("Loading data for training...")
        data_dir = Path(data_path)
        X_train = pd.read_pickle(data_dir / "X_train.pkl")
        y_train = pd.read_pickle(data_dir / "y_train.pkl").squeeze()

        # Ensure categorical features are properly set
        for col in cat_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category").cat.add_categories("missing").fillna("missing")

        logger.info(f"Training {model_name} model...")

        # Set up the model
        if model_name == 'catboost':
            model = CatBoostClassifier(
                random_seed=42,
                verbose=100,
                eval_metric='F1',
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
        run.log(metrics)

        # Save the trained model
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Log the model as a WandB artifact
        artifact = wandb.Artifact("trained_model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)

        logger.info("Training task completed successfully.")


@task
def predict_task(model_path: str, input_data_path: str, predictions_path: str) -> None:
    """Prediction Task: Generates predictions and logs results to WandB."""
    # Initialize WandB run for prediction
    with wandb.init(project="click-prediction", entity="wandb", job_type="prediction") as run:
        logger.info("Starting prediction...")
        
        # Run predictions
        predictions = predict(model_path=Path(model_path), input_data_path=Path(input_data_path))
        
        # Save predictions to a file
        predictions.to_csv(predictions_path, index=False)
        
        # Log predictions artifact to WandB
        artifact = wandb.Artifact("predictions", type="dataset")
        artifact.add_file(predictions_path)
        run.log_artifact(artifact)
        
        logger.info(f"Prediction completed. Results saved to {predictions_path}.")

#schedule = IntervalSchedule(interval=timedelta(days=1))

#@flow(name="Scheduled Workflow", schedule=schedule)
@flow
def workflow(input_path: str, processed_path: str, model_path: str, predictions_path: str) -> None:
    """Main Prefect Workflow: Orchestrates preprocessing, training, and prediction tasks."""
    preprocess_task(input_path=input_path, processed_path=processed_path)
    train_task(data_path=processed_path, model_path=model_path)
    predict_task(model_path=model_path, input_data_path=processed_path, predictions_path=predictions_path)
"""
from invoke import task
@task
def echo(c, echo_str):
	print(f"echo: {echo_str}")


@task 
def pipeline2(c):
    c.run("python preprocess.py")
    c.run("python train.py")
    c.run("python predict.py")

import subprocess
from prefect import task, flow

@task
def echo(echo_str):

    print(f"echo: {echo_str}")

@task
def run_script(script_name):

    subprocess.run(["python", script_name], check=True)
    print(f"Completed: {script_name}")

@flow
def pipeline2():

    echo("Starting the pipeline...")
    run_script("preprocess.py")  # Run preprocess script
    run_script("train.py")       # Run training script
    run_script("predict.py")     # Run prediction script
    echo("Pipeline completed!")
"""
# Example for running the workflow
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full Prefect workflow with WandB integration.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the raw input data.")
    parser.add_argument("--processed-path", type=str, required=True, help="Path to save the processed data.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--predictions-path", type=str, required=True, help="Path to save the predictions.")
    args = parser.parse_args()

    workflow(
        input_path=args.input_path,
        processed_path=args.processed_path,
        model_path=args.model_path,
        predictions_path=args.predictions_path,
    )
    
    

