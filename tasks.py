from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from preprocess import Preprocessor
from train import Trainer
from predict import Predictor
import pandas as pd
from pathlib import Path
# import wandb


# wandb_run = wandb.init(project="shay-kaggle-competition")


def load_data(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir)

if __name__ == '__main__':
    # Define paths
    data_path = "data/raw/train_dataset_full.csv"
    model_path = "models/model.pkl"
    result_path = "results/predictions.csv"

    data = load_data(data_path)

    # Split data
    train_data, validation_data = train_test_split(data, test_size=0.2, random_state=111)

    # Preprocessing
    preprocessor = Preprocessor()
    train_processed_data = preprocessor.preprocess_train(train_data)
    X_train, y_train = train_processed_data.drop(columns='is_click'), train_processed_data['is_click']

    # Training
    trainer = Trainer(X_train, y_train, n_trials=1, model_path=model_path)
    trainer.run_training()

    # Prediction
    pred_processed_data = preprocessor.preprocess_test(validation_data)
    X_val, y_val = pred_processed_data.drop(columns='is_click'), pred_processed_data['is_click']

    predictor = Predictor(model_path)
    predictions = predictor.predict(X_val)
    predictor.save_results(predictions, result_path)

    # Evaluate the result
    f1 = f1_score(y_val, predictions)
    print(f"F1 Score: {f1:.2f}")