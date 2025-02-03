from prefect import task, flow
import argparse
from pathlib import Path
from Preprocess_optional import DataPreprocessor
from train import ModelTrainer
import wandb
import json
import pandas as pd
import numpy as np

"""

+-+-+-+ +-+-+-+-+-+-+-+-+-+
|W|&|B| |C|a|l|l|b|a|c|k|s|
+-+-+-+ +-+-+-+-+-+-+-+-+-+

"""

def wandb_callback(metrics: dict):
    """
    Callback function for logging metrics to Weights & Biases (WandB).
    Ensures correct handling of train/validation F1 scores as line plots.
    """
    sample_size = 15000
    processed_metrics = {}

    train_f1_scores = []
    val_f1_scores = []
    folds = []

    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            df = value.copy().head(sample_size)
            for col in df.columns:
                if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype(str)
                    if "missing" in df[col].values:
                        df[col] = df[col].replace('missing', np.nan)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            processed_metrics[key] = wandb.Table(dataframe=df)

        elif isinstance(value, list) and "fold_scores_train" in key:
            train_f1_scores = value
        elif isinstance(value, list) and "fold_scores_val" in key:
            val_f1_scores = value
        elif isinstance(value, list) and "fold_numbers" in key:
            folds = value

        elif isinstance(value, pd.Series):
            processed_metrics[key] = wandb.Table(dataframe=value.to_frame())

        else:
            processed_metrics[key] = value

    # Ensure we have all required F1 scores before logging the plot
    if train_f1_scores and val_f1_scores and folds:
        chart = wandb.plot.line_series(
            xs=folds,
            ys=[train_f1_scores, val_f1_scores],
            keys=["Train F1", "Validation F1"],
            title="Train and Validation F1 Scores per Fold",
            xname="Fold"
        )
        wandb.log({"train_vs_val_f1_plot": chart})

    # Log everything else normally
    wandb.log(processed_metrics)



"""
+-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+
|P|r|e|p|r|o|c|e|s|s|i|n|g| |T|a|s|k|
+-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+

"""

@task(name="preprocess_data")
def preprocess_data(csv_path: str, output_path: str):
    preprocessor = DataPreprocessor(output_path=Path(output_path))
    df, X_test_1st = preprocessor.load_data(Path(csv_path))
    df_clean, X_train, X_test, y_train, y_test, fold_datasets, X_test_1st = preprocessor.preprocess(df, X_test_1st)
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test, fold_datasets, X_test_1st)
    return df_clean, X_train, X_test, y_train, y_test, fold_datasets, X_test_1st


"""
+-+-+-+-+ +-+-+-+-+
|T|u|n|e| |T|a|s|k|
+-+-+-+-+ +-+-+-+-+

"""

@task(name="tune_hyperparameters")
def tune_hyperparameters(trainer, folds_dir, n_trials, run_id):
    best_params = trainer.hyperparameter_tuning(
        X_train=None,  # Placeholder, actual data will be loaded inside the method
        y_train=None,  # Placeholder, actual data will be loaded inside the method
        cat_features=None,  # Placeholder, actual data will be loaded inside the method
        n_trials=n_trials,
        run_id=run_id
    )
    return best_params


"""
+-+-+-+-+-+-+-+-+ +-+-+-+-+
|T|r|a|i|n|i|n|g| |T|a|s|k|
+-+-+-+-+-+-+-+-+ +-+-+-+-+

"""

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

    # Log train and validation F1 scores for all folds in a single graph
    train_f1_scores = results["fold_scores_train"]
    val_f1_scores = results["fold_scores_val"]

    folds = list(range(1, len(train_f1_scores) + 1))
    data = [[fold, train, val] for fold, train, val in zip(folds, train_f1_scores, val_f1_scores)]
    table = wandb.Table(data=data, columns=["Fold", "Train F1", "Validation F1"])

# Create a line series plot with the fold numbers as x-axis and both F1 score lists as y-values.
    chart = wandb.plot.line_series(
        xs=table.get_column("Fold"),
        ys=[table.get_column("Train F1"), table.get_column("Validation F1")],
        keys=["Train F1", "Validation F1"],
        title="Train and Validation F1 Scores per Fold",
        xname="Fold"
    )

    # Log the chart to WandB.
    wandb.log({"train_val_f1_scores": chart})

    return results


"""
+-+-+-+-+ +-+-+-+-+
|M|a|i|n| |F|l|o|w|
+-+-+-+-+ +-+-+-+-+

"""

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
        config={
            "model_name": model_name,
            "n_trials": n_trials,
            "run_id": run_id
        }
    )

    if preprocess:
        preprocess_data(csv_path, output_path)

    base_trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        model_name=model_name,
        callback=wandb_callback,
    )

    best_params = None

    if tune:
        best_params = tune_hyperparameters(
            base_trainer, folds_dir, n_trials, run_id
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

        results = train_model(
            trainer_params=final_params_path,
            folds_dir=folds_dir,
            test_file=test_file,
            model_name=model_name,
            callback=wandb_callback,
            run_id=run_id
        )
        if results:
            print(f"✅ Training completed. Logging results: {results}")
            #wandb.log({"training_results": results})

    wandb.finish()



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

