import argparse
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def Preprocess(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    fillna: bool = False,
    output_path: Path = None,
    time_sensitive: bool = False,
    use_dummies: bool = True,
    save_as_pickle: bool = True
) -> pd.DataFrame:

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop duplicates
    df_clean = df.drop_duplicates().copy()
    logger.info(f"Initial shape: {df.shape}. After removing duplicates: {df_clean.shape}")

    # Convert DateTime column
    if 'DateTime' in df_clean.columns:
        df_clean['DateTime'] = pd.to_datetime(df_clean['DateTime'], errors='coerce')

    # Remove outliers
    if remove_outliers and 'age_level' in df_clean.columns:
        mean = df_clean['age_level'].mean()
        std = df_clean['age_level'].std()
        outlier_condition = (
            (df_clean["age_level"] > mean + 3 * std) |
            (df_clean["age_level"] < mean - 3 * std)
        )
        num_outliers = outlier_condition.sum()
        df_clean = df_clean[~outlier_condition]
        logger.info(f"Removed {num_outliers} outliers based on age_level")

    # Fill missing values
    if fillna:
        df_clean['product_category'] = df_clean['product_category_1'].fillna(
            df_clean['product_category_2']
        ).astype('category', errors='ignore')
        df_clean.drop(columns=['product_category_1', 'product_category_2'], inplace=True)

        cols_to_fill = ['is_click', 'product', 'gender', 'var_1', 'age_level']
        cols_to_fill = [col for col in cols_to_fill if col in df_clean.columns]

        df_clean[cols_to_fill] = (
            df_clean.groupby('user_id')[cols_to_fill]
            .transform(lambda x: x.ffill().bfill())
            .infer_objects(copy=False)
        )

        # Drop rows with too many missing values
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.dropna(thresh=len(df_clean.columns) - 5)
        logger.info(f"Dropped {initial_rows - df_clean.shape[0]} rows due to excessive missing values")
        #wildcard- remove rows with missing values in the 'is_click' column
        df_clean = df_clean.dropna(subset=['is_click'])

    # Convert categorical columns
    cat_cols = ['product_category', 'product', 'gender', 'campaign_id', 'webpage_id', 'user_group_id']
    for col in cat_cols:
        if col in df_clean.columns:
            if col in ['campaign_id', 'webpage_id', 'user_group_id', 'product_category']:
                df_clean[col] = df_clean[col].astype('Int64').astype('category')
            else:
                df_clean[col] = df_clean[col].astype('category')

    X = df_clean.drop(columns=['is_click'])
    y = df_clean['is_click']
    if time_sensitive:
        X.sort_values('DateTime', inplace=True)
        n = len(X)
        X_train = X.iloc[:int(np.ceil(0.8 * n))]
        X_test = X.iloc[int(np.ceil(0.8 * n)):] 
        y_train = y.iloc[:int(np.ceil(0.8 * n))]
        y_test = y.iloc[int(np.ceil(0.8 * n)):]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply get_dummies if use_dummies is True
    if use_dummies:
        dummied_group = ['product', 'product_category', 'gender', 'campaign_id', 'webpage_id', 'user_group_id']
        X_train = pd.get_dummies(X_train, columns=dummied_group, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=dummied_group, drop_first=True)

    # Save preprocessed data to specified output
    if output_path:
        output_dir = output_path.parent if output_path.suffix else output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_as_pickle:
            # Save as Pickle to preserve data types
            df_clean.to_pickle(output_dir / "cleaned_data_Maor.pkl")
            X_train.to_pickle(output_dir / "X_train.pkl")
            X_test.to_pickle(output_dir / "X_test.pkl")
            y_train.to_pickle(output_dir / "y_train.pkl")
            y_test.to_pickle(output_dir / "y_test.pkl")
            logger.info(f"Saved preprocessed data and splits as Pickle to {output_dir}")
        else:
            # Save as CSV for compatibility
            df_clean.to_csv(output_dir / "cleaned_data_Maor.csv", index=False)
            X_train.to_csv(output_dir / "X_train.csv", index=False)
            X_test.to_csv(output_dir / "X_test.csv", index=False)
            y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
            y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)
            logger.info(f"Saved preprocessed data and splits as CSV to {output_dir}")

    return df_clean, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save the output CSV file")
    parser.add_argument("--remove-outliers", action="store_true", help="Flag to remove outliers")
    parser.add_argument("--fillna", action="store_true", help="Flag to fill missing values")
    parser.add_argument("--use-dummies", action="store_true", help="Flag to apply pd.get_dummies")
    parser.add_argument("--save-as-pickle", action="store_true",default=True, help="Flag to save as Pickle instead of CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    logger.info(f"Loading file from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Preprocess the DataFrame with the passed arguments
    df_clean, _, _, _, _ = Preprocess(
        df,
        remove_outliers=args.remove_outliers,
        fillna=args.fillna,
        output_path=Path(args.output_path),
        use_dummies=args.use_dummies,
        save_as_pickle=args.save_as_pickle
    )
