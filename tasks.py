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
    remove_outliers: bool = False,
    fillna: bool = False,
    use_dummies: bool = True,
    save_as_pickle: bool = True
):
    """
    Task to preprocess data using the DataPreprocessor class.

    Args:
        csv_path (str): Path to the raw input CSV file.
        output_path (str): Path to save the preprocessed data.
        remove_outliers (bool): Flag to remove outliers during preprocessing.
        fillna (bool): Flag to fill missing values during preprocessing.
        use_dummies (bool): Flag to apply `pd.get_dummies` for categorical variables.
        save_as_pickle (bool): Flag to save outputs as Pickle instead of CSV.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting the preprocessing task...")

    # Initialize the DataPreprocessor
    preprocessor = DataPreprocessor(
        output_path=Path(output_path),
        remove_outliers=remove_outliers,
        fillna=fillna,
        use_dummies=use_dummies,
        save_as_pickle=save_as_pickle
    )

    # Load the raw data
    try:
        df = preprocessor.load_data(Path(csv_path))
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Perform preprocessing
    logger.info("Running the preprocessing pipeline...")
    df_clean, X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

    # Save preprocessed data
    logger.info("Saving preprocessed data...")
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test)

    logger.info("Preprocessing task completed successfully.")

@task
def train_task(data_path: str, model_path: str, cat_features: list, val_size: float = 0.2, model_name: str = "catboost") -> None:
    """
    Training Task: Trains a model, logs key metrics, and saves the trained model.

    Args:
        data_path (str): Path to the preprocessed data (pickled file containing X_train and y_train).
        model_path (str): Path to save the trained model.
        cat_features (list): List of categorical feature names or indices.
        val_size (float): Fraction of data to use for validation (default: 0.2).
        model_name (str): Name of the model to train (default: catboost).
    """
    logger.info("Loading data for training...")
    # Load the preprocessed data
    data_dir = Path(data_path)
    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    y_train = pd.read_pickle(data_dir / "y_train.pkl").squeeze()

    logger.info("Determining categorical features...")
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
    logger.info(f"Training the model with a validation set size of {val_size * 100:.1f}%...")
    model.fit(X_train_final, y_train_final, eval_set=(X_valid, y_valid), use_best_model=True)

    # Save the trained model
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")


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
    
    

