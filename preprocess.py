import argparse
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set pandas option to handle future behavior
pd.set_option('future.no_silent_downcasting', True)


def parse(csv_path: str) -> pd.DataFrame:
    """
    Preprocess the input CSV file.

    Args:
        csv_path (str): Path to the input CSV file

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    logger.info(f"Loading file from: {csv_path}")

    # Validate input path
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Load CSV to DataFrame
    df = pd.read_csv(csv_path)
    logger.info(f"Original shape: {df.shape}")

    # Store initial data quality metrics
    initial_metrics = {
        'total_rows': len(df),
        'null_counts': df.isnull().sum().to_dict()
    }

    # Preprocessing steps
    df_clean = (df
                .drop_duplicates(keep='first')
                .drop('product_category_2', axis=1))

    # Session deduplication
    df_clean = df_clean[~df_clean['session_id'].duplicated(keep='first') |
                        df_clean['session_id'].isna()]

    # Handle missing values
    df_clean.dropna(subset=['is_click'], axis=0, inplace=True)

    # Fill user-related features
    labels = ['gender', 'age_level', 'city_development_index', 'user_group_id']
    filled_features = (df_clean.groupby('user_id')[labels]
                       .transform(lambda x: x.ffill().bfill())
                       .infer_objects(copy=False))  # Add infer_objects
    df_clean[labels] = filled_features

    # Remove rows with too many missing values
    df_clean = df_clean.dropna(thresh=len(df_clean.columns) - 5)

    # Log preprocessing results
    logger.info(f"Final shape: {df_clean.shape}")
    logger.info(f"Removed {len(df) - len(df_clean)} rows during preprocessing")

    # Save preprocessed data
    output_path = Path('data/processed/cleaned_data.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed data to {output_path}")

    return df_clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess CTR prediction dataset')
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str,
                        default="data/processed/cleaned_data.csv",
                        help="Path to save the preprocessed data")
    args = parser.parse_args()

    try:
        df = parse(args.csv_path)
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


