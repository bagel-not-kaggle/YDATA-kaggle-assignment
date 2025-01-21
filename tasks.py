from prefect import task, flow
from pathlib import Path
from preprocess import Preprocess  # Assuming Preprocess is the main function in preprocess.py
from train import train_model  # Assuming train_model is the main function in train.py
from predict import predict  # Assuming predict is the main function in predict.py
import logging
#from prefect.schedules import IntervalSchedule
from datetime import timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def preprocess_task(input_path: str, processed_path: str, use_dummies: bool = False, save_as_pickle: bool = True) -> None:
    """
    Wrapper for the preprocessing function.
    """
    logger.info("Starting preprocessing...")
    df = Preprocess(
        df=pd.read_csv(input_path),
        output_path=Path(processed_path),
        use_dummies=use_dummies,
        save_as_pickle=save_as_pickle,
    )
    logger.info(f"Preprocessing completed. Data saved to {processed_path}.")

@task
def train_task(data_path: str, model_path: str) -> None:
    """
    Wrapper for the training function.
    """
    logger.info("Starting training...")
    train_model(data_path, model_path)
    logger.info(f"Training completed. Model saved to {model_path}.")

@task
def predict_task(model_path: str, input_data_path: str, predictions_path: str) -> None:
    """
    Wrapper for the prediction function.
    """
    logger.info("Starting prediction...")
    predict(model_path=Path(model_path), input_data_path=Path(input_data_path), output_path=Path(predictions_path))
    logger.info(f"Prediction completed. Results saved to {predictions_path}.")

#schedule = IntervalSchedule(interval=timedelta(days=1))

#@flow(name="Scheduled Workflow", schedule=schedule)
@flow
def workflow(input_path: str, processed_path: str, model_path: str, predictions_path: str) -> None:
    """
    Define the workflow that chains all tasks together.
    """
    preprocess_task(input_path=input_path, processed_path=processed_path)
    train_task(data_path=processed_path, model_path=model_path)
    predict_task(model_path=model_path, input_data_path=processed_path, predictions_path=predictions_path)

# Example for running the workflow
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full Prefect workflow.")
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
