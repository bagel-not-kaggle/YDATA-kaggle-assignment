from pathlib import Path
import numpy as np
import pandas as pd
import argparse

def load_data(data_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(data_dir)
    return data

def drop_nan(data: pd.DataFrame, selected_columns: str) -> pd.DataFrame:
    mask = data[selected_columns].notna().all(axis=1)
    return data[mask]


if __name__ == '__main__':
    data_dir = 'data/raw'
    file_name = 'train_dataset_full.csv'
    data_path = Path(f"{data_dir}/{file_name}")
    train_dataset = load_data(data_path)
    print(train_dataset.shape)

    # Remove rows with NaN values
    selected_columns = ['session_id', 'DateTime', 'user_id', 'product', 'campaign_id', 'webpage_id', 'is_click']
    filtered_dataset = drop_nan(train_dataset, selected_columns)
    print(filtered_dataset.shape)