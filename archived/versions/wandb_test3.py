import wandb
from prefect import task, flow
from pathlib import Path
from archived.versions.preprocess_cat import DataPreprocessor
import argparse
import pandas as pd
import numpy as np

def wandb_callback(metrics: dict):
    processed_metrics = {}
    sample_size = 15000
    
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            df = value.copy().head(sample_size)
            for col in df.columns:
                #if pd.api.types.is_categorical_dtype(df[col]):
                 #   df[col] = df[col].cat.add_categories([np.nan]).fillna(np.nan)
                #elif df[col].dtype == 'object':
                df[col] = df[col].replace('missing', np.nan)
            processed_metrics[key] = wandb.Table(dataframe=df)
        elif isinstance(value, pd.Series):
            series = value.copy().head(sample_size)
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str)
            elif series.dtype == 'object':
                series = series.replace('missing', np.nan)
            processed_metrics[key] = wandb.Table(dataframe=series.to_frame())
    
    wandb.log(processed_metrics)




@flow(name="preprocess_data_flow")
def preprocess_flow(csv_path: str, output_path: str):
    wandb.init(
        project="ctr-prediction",
        settings=wandb.Settings(start_method="thread"),
        config={
            "remove_outliers": False,
            "fillna": True,
            "use_dummies": False,
            "save_as_pickle": True,
            "n_folds": 5
        }
    )
    
    preprocessor = DataPreprocessor(
        output_path=Path(output_path),
        remove_outliers=wandb.config.remove_outliers,
        fillna=wandb.config.fillna,
        use_dummies=wandb.config.use_dummies,
        save_as_pickle=wandb.config.save_as_pickle,
        callback=wandb_callback
    )
    
    df = preprocessor.load_data(Path(csv_path))
    df_clean, X_train, X_test, y_train, y_test, fold_datasets = preprocessor.preprocess(df)
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test, fold_datasets)
    
    wandb.finish()

if __name__ == "__main__":
    preprocess_flow("data/raw/train_dataset_full.csv", "data/processed")
