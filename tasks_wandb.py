from pathlib import Path
from prefect import task, flow
from Preprocess_optional import DataPreprocessor
from train import ModelTrainer
import wandb
import json
import pandas as pd
import argparse
import numpy as np

#########################################
#          Wandbak callback             #
#########################################


def wandb_callback(metrics: dict):
    """Handles all DataFrame/Series conversion scenarios"""
    
    def process_value(value):
        # Handle DataFrames
        if isinstance(value, pd.DataFrame):
            df = value.head(5000).copy()
            # Force numeric columns (campaign_id etc.) to stay numeric
            num_cols = ["campaign_id", "webpage_id", "user_group_id"]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].replace("missing", np.nan), errors='coerce')
            return wandb.Table(dataframe=df)
        
        # Handle Series
        elif isinstance(value, pd.Series):
            return wandb.Table(dataframe=value.to_frame())
        
        # Handle lists/tuples containing DataFrames
        elif isinstance(value, (list, tuple)):
            return [process_value(v) for v in value]
        
        # Handle dictionaries (like fold_datasets metadata)
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        
        # Pass through other types
        else:
            return value
    
    processed_metrics = {k: process_value(v) for k, v in metrics.items()}
    wandb.log(processed_metrics)

#########################################
#         Preprocessing Task            #
#########################################

@task(name="preprocess_data")
def preprocess_data(csv_path: str, output_path: str, callback=None):
    preprocessor = DataPreprocessor(output_path=Path(output_path))
    df, X_test_1st = preprocessor.load_data(Path(csv_path))
    
    # Preprocessing
    df_clean, X_train, X_test, y_train, y_test, fold_datasets, X_test_1st = preprocessor.preprocess(df, X_test_1st)

    # Debug prints
    print("✅ After preprocessing:")
    print("df_clean columns:", df_clean.columns)
    print("X_train columns:", X_train.columns)
    print("y_train shape:", y_train.shape)
    print("y_train sample:", y_train.head())

    
    # Save Data
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test, fold_datasets, X_test_1st)
    
    # Confirm Folds Are Correct
    if fold_datasets:
        print("Folds length:", len(fold_datasets))
        print("First fold y_train sample:", fold_datasets[0][1].head())

    # Callback to W&B (using stage 'preprocess' to force table conversion)
    if callback:
        callback({
            "df_clean": df_clean,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "fold_datasets": fold_datasets,
            "X_test_1st": X_test_1st
        })
    
    return df_clean, X_train, X_test, y_train, y_test, fold_datasets, X_test_1st

#########################################
#        Hyperparameter Tuning Task     #
#########################################

@task(name="tune_hyperparameters")
def tune_hyperparameters(trainer, folds_dir, n_trials, run_id):
    # Load training data
    X_train = pd.read_pickle(trainer.folds_dir / "X_train.pkl")
    y_train = pd.read_pickle(trainer.folds_dir / "y_train.pkl").squeeze()
    cat_features = trainer.determine_categorical_features(X_train)

    best_params = trainer.hyperparameter_tuning(
        X_train=X_train,
        y_train=y_train,
        cat_features=cat_features,
        n_trials=n_trials,
        run_id=run_id
    )
    return best_params

#########################################
#           Training Task               #
#########################################

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
        params=trainer_params
    )
    results = trainer.train_and_evaluate()

    # Log train and validation F1 scores for all folds
    for fold_index, (train_f1, val_f1) in enumerate(zip(results["fold_scores_train"], results["fold_scores_val"])):
        wandb.log({
            f"Fold {fold_index + 1} Train F1": train_f1,
            f"Fold {fold_index + 1} Validation F1": val_f1
        })

    # Create a single plot with two line plots
    data = [[fold, train_f1, val_f1] for fold, (train_f1, val_f1) in enumerate(zip(results["fold_scores_train"], results["fold_scores_val"]), 1)]
    table = wandb.Table(data=data, columns=["Fold", "Train F1", "Validation F1"])

    wandb.log({
        "train_val_f1_scores": wandb.plot.line_series(
            xs=table.get_column("Fold"),
            ys=[table.get_column("Train F1"), table.get_column("Validation F1")],
            keys=["Train F1", "Validation F1"],
            title="Train and Validation F1 Scores per Fold",
            xname="Fold"
        )
    })

    # Log average F1 scores
    wandb.log({
        "Average Train F1": results["avg_f1_train"],
        "Average Validation F1": results["avg_f1_val"],
        "Best Validation F1": results["best_f1"],
        "Test F1": results["test_f1"]
    })

    return results

#########################################
#             Main Flow                 #
#########################################

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
    wandb.init(
        project="ctr-prediction",
        settings=wandb.Settings(start_method="thread"),
        name=f"{model_name}_run_{run_id}",
        config={
            "model_name": model_name,
            "n_trials": n_trials,
            "run_id": run_id
        }
    )

    if preprocess:
        preprocess_data(csv_path, output_path, callback=wandb_callback)

    base_trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        model_name=model_name,
        callback=wandb_callback,
    )

    best_params = None

    if tune:
        best_params = tune_hyperparameters(
            trainer=base_trainer, folds_dir=folds_dir, n_trials=n_trials, run_id=run_id
        )
        
        if best_params:
            wandb.config.update(best_params)
            best_params_path = f"data/Hyperparams/best_params{run_id}.json"
            with open(best_params_path, "w") as f:
                json.dump(best_params, f, indent=4)
            print(f"✅ Saved best hyperparameters to {best_params_path}")

    if train:
        final_params_path = params if params else f"data/Hyperparams/best_params{run_id}.json"
        print(f"✅ Using hyperparameter file: {final_params_path}")

        train_model(
            trainer_params=final_params_path,
            folds_dir=folds_dir,
            test_file=test_file,
            model_name=model_name,
            callback=wandb_callback,
            run_id=run_id
        )

    wandb.finish()

#########################################
#         CLI Argument Parsing          #
#########################################

if __name__ == "__main__":
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

    args = parser.parse_args()

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
