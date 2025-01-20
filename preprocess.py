import argparse
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    fillna: bool = False,
    output_path: Path = Path("data/processed/cleaned_data.csv")
) -> pd.DataFrame:
    # Drop duplicates
    df = df.copy()  # Ensure the original DataFrame is not modified
    df_clean = df.drop_duplicates()
    logger.info(f"Initial shape: {df.shape}. After removing duplicates: {df_clean.shape}")

    # Convert DateTime column
    if 'DateTime' in df_clean.columns:
        df_clean.loc[:, 'DateTime'] = pd.to_datetime(df_clean['DateTime'], errors='coerce')

    # Convert certain columns to categorical
    cat_cols = ['product_category_1', 'is_click', 'product_category_2', 'product', 'gender', 'var_1']


    # Remove outliers
    if remove_outliers and 'age_level' in df_clean.columns:
        mean = df_clean['age_level'].mean()
        std = df_clean['age_level'].std()
        outlier_condition = (
            (df_clean["age_level"] > mean + 3 * std) |
            (df_clean["age_level"] < mean - 3 * std)
        )
        outliers = df_clean[outlier_condition]
        df_clean = df_clean.drop(outliers.index)
        logger.info(f"Removed {len(outliers)} outliers based on age_level")

    # Fill missing values
    if fillna:
        df_clean.loc[:, 'product_category_1'] = df_clean['product_category_1'].astype(str)
        df_clean.loc[:, 'product_category_2'] = df_clean['product_category_2'].astype(str)

        # Fill missing values and combine into a single column
        df_clean['product_category'] = df_clean['product_category_1'].fillna(df_clean['product_category_2'])

        # Drop the original columns
        df_clean.drop(columns=['product_category_1', 'product_category_2'], inplace=True)

        # Convert the combined column to category after fillna
        df_clean.loc[:, 'product_category'] = df_clean['product_category'].astype('category')

        cols_to_fill = ['product_category', 'is_click', 'product', 'gender', 'var_1', 'age_level']
        cols_to_fill = [col for col in cols_to_fill if col in df_clean.columns]
        df_clean[cols_to_fill] = (
        df_clean.groupby('user_id')[cols_to_fill]
        .transform(lambda x: x.ffill().bfill())
        .infer_objects(copy=False)  # Explicitly opt-in to inferred types
            )

        # Drop rows with too many missing values
        df_clean = df_clean.dropna(thresh=len(df_clean.columns) - 5)
        logger.info(f"Final shape after fillna and dropping rows: {df_clean.shape}")

    for col in cat_cols:
        if col in df_clean.columns:
            df_clean.loc[:, col] = df_clean[col].astype(str).astype('category')
    # Save preprocessed data to specified output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed data to {output_path}")

    return df_clean



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, default="data/processed/cleaned_data.csv", help="Path to save the output CSV file")
    parser.add_argument("--remove-outliers", action="store_true", help="Flag to remove outliers")
    parser.add_argument("--fillna", action="store_true", help="Flag to fill missing values")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    logger.info(f"Loading file from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Preprocess the DataFrame with the passed output path
    df_clean = preprocess(
        df,
        remove_outliers=args.remove_outliers,
        fillna=args.fillna,
        output_path=Path(args.output_path)  # <-- Provide output path here
    )

    logger.info(f"Preprocessed data saved to {args.output_path}")

