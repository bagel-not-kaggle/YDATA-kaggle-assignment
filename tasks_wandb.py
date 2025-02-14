import wandb
from prefect import task, flow
from pathlib import Path
from preprocess import DataPreprocessor
from train import ModelTrainer
import pandas as pd
import numpy as np
import json
import argparse
import holoviews as hv
from catboost import CatBoostClassifier
import panel as pn
from bokeh.resources import INLINE
hv.extension("bokeh", logo=False)
from error_analysis import error_analysis

"""
+-+-+-+ +-+-+-+-+-+-+-+-+-+
|W|&|B| |C|a|l|l|b|a|c|k|s|
+-+-+-+ +-+-+-+-+-+-+-+-+-+
"""

def wandb_callback(metrics: dict):
    """
    Handles DataFrame/Series conversion and ensures W&B logs correctly.
    """
    if "mean_PRAUC" in metrics and "trial_number" in metrics:
        wandb.log({
            "Hyperparameter Tuning/Mean PRAUC": metrics["mean_PRAUC"],
            "trial": metrics["trial_number"]
        })
    processed_metrics = {}
    sample_size = 15000

    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            #shuffle rows
            df = value.sample(frac=1).head(sample_size).copy()
            df = df.astype(str)  # Convert everything to string to avoid serialization issues
            processed_metrics[key] = wandb.Table(dataframe=df)

        elif isinstance(value, pd.Series):
            series_df = value.to_frame().head(sample_size).astype(str)
            processed_metrics[key] = wandb.Table(dataframe=series_df)

        elif isinstance(value, (int, float, str, dict)):
            processed_metrics[key] = value  # Log scalars and dictionaries directly

    wandb.log(processed_metrics)

"""
+-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+
|P|r|e|p|r|o|c|e|s|s|i|n|g| |T|a|s|k|
+-+-+-+-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+
"""

@task(name="preprocess_data")
def preprocess_data(csv_path: str, output_path: str,test_path:str):
    """
    Load raw data, run preprocessing, and save the processed data. Logs essential details to W&B.
    """
    preprocessor = DataPreprocessor(
        output_path=Path(output_path),
        remove_outliers=False,
        fillna=True,
        use_dummies=False,
        save_as_pickle=True,
        callback=wandb_callback
    )
    df, X_test_1st = preprocessor.load_data(Path(csv_path), Path(test_path))
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
    
"""
+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
|F|e|a|t|u|r|e| |S|e|l|e|c|t|i|o|n|
+-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+

"""


@task(name="feature_selection")
def feature_select(trainer, n_trials, run_id, folds_dir):
    model = CatBoostClassifier()
    model_path = Path('models') / 'best_model_catboost.cbm'
    model.load_model(str(model_path))
    
    # Get feature importance and print column names for verification
    feature_importance = model.get_feature_importance(prettified=True)
    print("Available columns:", feature_importance.columns)
    
    # Use correct column names from CatBoost output
    feature_names = feature_importance.iloc[:, 0].tolist()  # First column for feature names
    importance_values = feature_importance.iloc[:, 1].astype(float).tolist()  # Second column for importance values
    
    # Create visualization data
    feature_data = list(zip(feature_names, importance_values))
    feature_data.sort(key=lambda x: x[1], reverse=True)
    sorted_features, sorted_importance = zip(*feature_data)
    
    # Create holoviews bar plot
    bars = hv.Bars((sorted_features, sorted_importance), 'Features', 'Importance')
    bars.opts(
        width=800,
        height=400,
        tools=['hover'],
        title='Feature Importance Rankings',
        xrotation=45
    )
    
    html_file_name = "feature_importance.html"
    pn.pane.HoloViews(bars).save(html_file_name)
    
    wandb_html = wandb.Html(html_file_name)
    table = wandb.Table(columns=["Feature_Importance_Plot"], data=[[wandb_html]])
    
    wandb.log({
        "Feature_Importance_Table": table,
        "Selected Features Count": len(feature_names),
        "Selected Features": wandb.Table(
            data=[[feat] for feat in feature_names],
            columns=["Feature"]
        )
    })
    
    return feature_names, importance_values, feature_names







"""
+-+-+-+-+-+-+-+-+ +-+-+-+-+
|T|r|a|i|n|i|n|g| |T|a|s|k|
+-+-+-+-+-+-+-+-+ +-+-+-+-+
"""

@task(name="train_model")
def train_model(trainer_params, folds_dir, test_file, model_name, callback,
                 run_id, features_path=None, select_features=False):
    """
    Load best hyperparams from JSON (assuming it was saved by the tuner), then train and evaluate the model.
    """
    trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        model_name=model_name,
        callback=callback,
        params=trainer_params,
        select_features=select_features,
        features_path= features_path
    )
    results = trainer.train_and_evaluate()
    
    # Log train and validation PRAUC scores per fold
    for fold_index, (train_prauc, val_prauc) in enumerate(zip(results["fold_scores_train"], results["fold_scores_val"])):
        wandb.log({
            f"Fold {fold_index + 1} Train PRAUC": train_prauc,
            f"Fold {fold_index + 1} Validation PRAUC": val_prauc
        })
    
    # Create a train/validation PRAUC plot
    data = [[fold, train_prauc, val_prauc] for fold, (train_prauc, val_prauc) in enumerate(zip(results["fold_scores_train"], results["fold_scores_val"]), 1)]
    table = wandb.Table(data=data, columns=["Fold", "Train PRAUC", "Validation PRAUC"])
    
    wandb.log({
        "train_val_PRAUC_scores": wandb.plot.line_series(
            xs=table.get_column("Fold"),
            ys=[table.get_column("Train PRAUC"), table.get_column("Validation PRAUC")],
            keys=["Train PRAUC", "Validation PRAUC"],
            title="Train and Validation PRAUC Scores per Fold",
            xname="Fold"
        )
    })
    
    return results

@task(name="Error Analysis")
def error_analyze(categorical_columns=['product', 'campaign_id', 'user_group_id', 'age_level', 'user_depth', 'city_development_index']):
    analyzer = error_analysis()
    df, ece = analyzer.compute_final_df()
    
    # Save the error analysis data
    df.to_csv('data/Predictions/error_analysis.csv', index=False)
    
    # Create and save the interactive visualization
    analyzer.create_interactive_plot(df, categorical_columns)
    
    # Similar to feature selection task, log the HTML to wandb
    html_file_name = "error_analysis.html"
    wandb_html = wandb.Html(html_file_name)
    table = wandb.Table(columns=["Error_Analysis_Plot"], data=[[wandb_html]])
    
    # Log metrics and visualizations to wandb
    wandb.log({
        "Error_Analysis_Visualization": table,
        "Expected_Calibration_Error": ece,
        "Error_Analysis_Data": wandb.Table(dataframe=df)
    })
    
    return df, ece

    
"""
+-+-+-+-+ +-+-+-+-+
|M|a|i|n| |F|l|o|w|
+-+-+-+-+ +-+-+-+-+
"""

@flow(name="preprocess_and_train_flow")
def preprocess_and_train_flow(
    csv_path: str = "data/raw/train_dataset_full.csv",
    test_path: str = "data/raw/X_test_1st.csv",
    output_path: str = "data/processed",
    folds_dir: str = "data/processed",
    test_file: str = "data/processed",
    model_name: str = "catboost",
    run_id: str = "1",
    n_trials: int = 50,
    preprocess: bool = False,
    tune: bool = False,
    best_features: bool = False,
    select_features: bool = False,
    analyze_errors: bool = False,
    train: bool = False,
    params=None
):
    """
    High-level Prefect flow that:
      1. Initializes a W&B run.
      2. Preprocesses the data.
      3. Optionally tunes hyperparameters.
      4. Optionally trains/evaluates a final model.
      5. Finishes the W&B run.
    """
    wandb.init(
        project="ctr-prediction",
        settings=wandb.Settings(start_method="thread"),
        config={"model_name": model_name, "n_trials": n_trials, "run_id": run_id}
    )

    if preprocess:
        preprocess_data(csv_path, output_path, test_path)
    
    base_trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        model_name=model_name,
        callback=wandb_callback,
        params= params,
        select_features=select_features
    )
    
    best_params = None
    best_params_path = params
    if tune:
        best_params = tune_hyperparameters(base_trainer, folds_dir, n_trials, run_id)
        
        best_params_path = f'data/Hyperparams/best_params{run_id}.json'
        wandb.config.update(best_params)
        base_trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        model_name=model_name,
        callback=wandb_callback,
        params=best_params_path
    )

    
    
    if best_features:
        # Run feature selection and wait for results
        best_features, feature_importance, feature_names = feature_select(base_trainer, n_trials, run_id, folds_dir)
        wandb.config.update({"selected_features": list(best_features)})


        

    if train:
        if best_params is None and params:  # Load from JSON if not tuning now
            with open(params, "r") as f:
                best_params = json.load(f)
        if select_features:
            features_path = f'data/Hyperparams/best_features{run_id}.pkl'

        train_model(best_params_path, folds_dir, test_file, 
                    model_name, wandb_callback, run_id, features_path=features_path, select_features=select_features)

    if analyze_errors:
        error_analyze()

    wandb.finish()


#########################################
#         CLI Argument Parsing          #
#########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocess_and_train_flow with optional parameters.")
    parser.add_argument("--csv_path", type=str, default="data/raw/train_dataset_full.csv", help="Path to the raw CSV file.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save the processed data.")
    parser.add_argument("--test_path", type=str, default="data/raw/test.csv", help="Path to test file")
    parser.add_argument("--folds_dir", type=str, default="data/processed", help="Directory for folds.")
    parser.add_argument("--test_file", type=str, default="data/processed", help="Path to the test file.")
    parser.add_argument("--model_name", type=str, default="catboost", help="Name of the model.")
    parser.add_argument("--run_id", type=str, default="1", help="Run ID.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of hyperparameter tuning trials.")
    parser.add_argument("--preprocess", action='store_true', help="Run preprocessing step.")
    parser.add_argument("--tune", action='store_true', help="Run hyperparameter tuning step.")
    parser.add_argument("--best_features", action='store_true', help="Run feature selection step.")
    parser.add_argument("--select_features", action='store_true', help="Run feature selection step.")
    parser.add_argument("--train", action='store_true', help="Run training step.")
    parser.add_argument("--analyze_errors", action='store_true', help="Run error analysis step.")
    parser.add_argument("--params", type=str, default=None, help="Path to the best hyperparameters JSON file.")

    args = parser.parse_args()

    preprocess_and_train_flow(
        csv_path=args.csv_path,
        output_path=args.output_path,
        test_path=args.test_path,
        folds_dir=args.folds_dir,
        test_file=args.test_file,
        model_name=args.model_name,
        run_id=args.run_id,
        n_trials=args.n_trials,
        preprocess=args.preprocess,
        tune=args.tune,
        best_features=args.best_features,
        select_features= args.select_features,
        analyze_errors=args.analyze_errors,
        train=args.train,
        params=args.params
    )