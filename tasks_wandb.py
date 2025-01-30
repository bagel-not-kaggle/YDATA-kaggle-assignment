import wandb
from prefect import task, flow
from pathlib import Path
from preprocess_cat import DataPreprocessor
from train import ModelTrainer
import pandas as pd
import numpy as np
import json

##############################################################################
# 1. Universal W&B Callback
##############################################################################
def wandb_callback(metrics: dict):
    """
    A single callback that can be used for both preprocessing and training.
    It checks if `metrics[key]` is a DataFrame, Series, or something else,
    and logs appropriately to Weights & Biases.
    """
    sample_size = 15000
    processed_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            df = value.copy().head(sample_size)
            # Example transformations for DF logging
            for col in df.columns:
                if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype(str)
                    if "missing" in df[col].values:
                        df[col] = df[col].replace('missing', np.nan)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            processed_metrics[key] = wandb.Table(dataframe=df)

            # If this is a trial_metrics DataFrame, log additional items
            if key == "trial_metrics" and "trial_number" in df.columns and "mean_f1_score" in df.columns:
                wandb.log({
                    "trial_number": df["trial_number"].iloc[0],
                    "mean_f1_score": df["mean_f1_score"].iloc[0]
                })

        elif isinstance(value, pd.Series):
            # Convert Series to DataFrame for logging
            series_df = value.to_frame()
            processed_metrics[key] = wandb.Table(dataframe=series_df)
        
        else:
            # If it's just a scalar or dictionary
            processed_metrics[key] = value
    
    # Log everything together
    wandb.log(processed_metrics)

##############################################################################
# 2. Preprocessing Task
##############################################################################
@task(name="preprocess_data")
def preprocess_data(csv_path: str, output_path: str):
    """
    Load raw data, run preprocessing, and save the processed data. Uses the
    same wandb_callback for logging.
    """
    preprocessor = DataPreprocessor(
        output_path=Path(output_path),
        remove_outliers=False,
        fillna=True,
        use_dummies=False,
        save_as_pickle=True,
        callback=wandb_callback
    )
    df = preprocessor.load_data(Path(csv_path))
    df_clean, X_train, X_test, y_train, y_test, fold_datasets = preprocessor.preprocess(df)
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test, fold_datasets)

##############################################################################
# 3. Tuning Task
##############################################################################
@task(name="tune_hyperparameters")
def tune_hyperparameters(trainer, folds_dir, n_trials, run_id):
    X_train = pd.read_pickle(Path(folds_dir) / "X_train.pkl")
    y_train = pd.read_pickle(Path(folds_dir) / "y_train.pkl").squeeze()
    cat_features = trainer.determine_categorical_features(X_train)
    
    return trainer.hyperparameter_tuning(
        X_train=X_train,
        y_train=y_train,
        cat_features=cat_features,
        n_trials=n_trials,
        run_id=run_id
    )

##############################################################################
# 4. Training Task
##############################################################################
@task(name="train_model")
def train_model(trainer_params, folds_dir, test_file, model_name, callback, run_id):
    """
    Load best hyperparams from JSON (assuming it was saved by the tuner),
    then train and evaluate the model.
    """
    
    
    trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        model_name=model_name,
        callback=callback,
        params = trainer_params
    )
    return trainer.train_and_evaluate()

##############################################################################
# 5. Main Flow
##############################################################################
@flow(name="preprocess_and_train_flow")
def preprocess_and_train_flow(
    csv_path: str = "data/raw/train_dataset_full.csv",
    output_path: str = "data/processed",
    folds_dir: str = "data/processed",
    test_file: str = "data/processed",
    model_name: str = "catboost",
    run_id: str = "1",
    n_trials: int = 50,
    preprocess: bool = False,
    tune: bool = False,
    train: bool = False,
    params = None
):
    """
    High-level Prefect flow that:
      1. Initializes a W&B run.
      2. Preprocesses the data.
      3. Optionally tunes hyperparameters.
      4. Optionally trains/evaluates a final model.
      5. Finishes the W&B run.
    """
    # Initialize a WandB run for the entire pipeline
    wandb.init(
        project="ctr-prediction",
        settings=wandb.Settings(start_method="thread"),
        config={
            "model_name": model_name,
            "n_trials": n_trials,
            "run_id": run_id
        }
    )
    # Step 1: Preprocess
    if preprocess:

        preprocess_data(csv_path, output_path)

    # Step 2: Create a base trainer for potential tuning or training
    base_trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        model_name=model_name,
        callback=wandb_callback,
    )
    
    # Step 3: Tune (optional)
    best_params = None
    if tune:
        best_params = tune_hyperparameters(
            base_trainer, folds_dir, n_trials, run_id
        ).result()
        if best_params:
            wandb.config.update(best_params)

    # Step 4: Train (optional)

    loaded_params = None

    # Load hyperparameters if a params file is provided via CLI
    if params is not None:
        try:
            with open(params, 'r') as f:
                loaded_params = json.load(f)
            print(f"Loaded hyperparameters from {params}: {loaded_params}")
        except Exception as e:
            print(f"Error loading hyperparameters from {params}: {e}")
    if train:
        final_params_path = params if params else f"data/Hyperparams/best_params{run_id}.json"

        print(f"✅ Using hyperparameter file: {final_params_path}")

        train_results = train_model(
            trainer_params=final_params_path,  # ✅ Pass the file path, NOT the dictionary
            folds_dir=folds_dir,
            test_file=test_file,
            model_name=model_name,
            callback=wandb_callback,
            run_id=run_id
        )

    
    # Finish the W&B run
    wandb.finish()

##############################################################################
# 6. Command-line entry point
##############################################################################
#if __name__ == "__main__":
    # Example default call
    #preprocess_and_train_flow(
    #    csv_path="data/raw/train_dataset_full.csv",
    #    output_path="data/processed",
    #    folds_dir="data/processed",
    #    test_file="data/processed",
    #    model_name="catboost",
    #    run_id="1",
    #    n_trials=50,
    #    params = None,
    #    preprocess=False,
    #    tune=False,
    #    train=True
    #)
import argparse
from prefect import flow




if __name__ == "__main__":
    # Argument parser for local execution
    parser = argparse.ArgumentParser(description="Run the preprocess_and_train_flow with optional parameters.")
    parser.add_argument("--csv_path", type=str, default="data/raw/train_dataset_full.csv", help="Path to the raw CSV file.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save the processed data.")
    parser.add_argument("--folds_dir", type=str, default="data/processed", help="Directory for folds.")
    parser.add_argument("--test_file", type=str, default="data/processed", help="Path to the test file.")
    parser.add_argument("--model_name", type=str, default="catboost", help="Name of the model.")
    parser.add_argument("--run_id", type=str, default="1", help="Run ID.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of hyperparameter tuning trials.")
    parser.add_argument("--preprocess", action='store_true', help="Run preprocessing step.")
    parser.add_argument("--tune", action='store_true', help="Run hyperparameter tuning step.")
    parser.add_argument("--train", action='store_true', help="Run training step.")
    parser.add_argument("--params", type=str, default=None, help="Path to JSON file with preloaded parameters.")

    # Parse arguments
    args = parser.parse_args()

    # Call the Prefect flow with parsed arguments
    preprocess_and_train_flow(
        csv_path=args.csv_path,
        output_path=args.output_path,
        folds_dir=args.folds_dir,
        test_file=args.test_file,
        model_name=args.model_name,
        run_id=args.run_id,
        n_trials=args.n_trials,
        preprocess=args.preprocess,
        tune=args.tune,
        train=args.train,
        params=args.params
    )

