import argparse
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import numpy as np

class DataPreprocessor:
    def __init__(
        self,
        output_path: Path = Path("data/processed"),
        remove_outliers: bool = False,
        fillna: bool = False,
        use_dummies: bool = True,
        save_as_pickle: bool = True,
        time_sensitive: bool = False
    ):
        self.output_path = output_path
        self.remove_outliers = remove_outliers
        self.fillna = fillna
        self.use_dummies = use_dummies
        self.save_as_pickle = save_as_pickle
        self.time_sensitive = time_sensitive

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        self.logger.info(f"Loading file from: {csv_path}")
        return pd.read_csv(csv_path)

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if "age_level" in df.columns:
            mean = df["age_level"].mean()
            std = df["age_level"].std()
            outlier_condition = (
                (df["age_level"] > mean + 3 * std) |
                (df["age_level"] < mean - 3 * std)
            )
            num_outliers = outlier_condition.sum()
            df = df[~outlier_condition]
            self.logger.info(f"Removed {num_outliers} outliers based on age_level")
        return df
    
    def fill_missing_with_mode(df: pd.DataFrame, columns: list):

        for column in columns:
            if column in df.columns:
                mode_value = df[column].mode()[0]  # Calculate the mode
                df[column] = df[column].fillna(mode_value)  # Fill missing values with the mode
        return df
    
    def fill_missing_with_median(df: pd.DataFrame, columns: list):

        for column in columns:
            if column in df.columns:
                median_value = df[column].median()  # Calculate the median
                df[column] = df[column].fillna(median_value)  # Fill missing values with the median
        return df 

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if "product_category_1" in df.columns and "product_category_2" in df.columns:
            df["product_category"] = df["product_category_1"].fillna(
                df["product_category_2"]
            ).astype("category", errors="ignore")
            df.drop(columns=["product_category_1", "product_category_2"], inplace=True)

        cols_to_fill = ["is_click", "product", "gender", "var_1", "age_level"]
        cols_to_fill = [col for col in cols_to_fill if col in df.columns]

        df[cols_to_fill] = (
            df.groupby("user_id")[cols_to_fill]
            .transform(lambda x: x.ffill().bfill())
            .infer_objects(copy=False)
        )

        initial_rows = df.shape[0]
        df = df.dropna(thresh=len(df.columns) - 5)

        self.logger.info(f"Dropped {initial_rows - df.shape[0]} rows due to excessive missing values")
        cols_to_fill_with_mode = ["product", "campaign_id", "webpage_id", "user_group_id", "gender", "age_level", "user_depth", "city_development_index", "var_1", "product_category",
                     "month", "day", "hour"]
      # cols_to_fill_with_median = ["Month", "Day", "Hour", "Minute", "weekday"] cols not created yets
        df = self.fill_missing_with_mode(df, cols_to_fill_with_mode)

        df = df.dropna(subset=["is_click"])
        return df
    

    def convert_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = ["product_category", "product", "gender", "campaign_id", "webpage_id", "user_group_id"]
        for col in cat_cols:
            if col in df.columns:
                if col in ["campaign_id", "webpage_id", "user_group_id", "product_category"]:
                    df[col] = df[col].astype("Int64").astype("category")
                else:
                    df[col] = df[col].astype("category")
        return df

    def split_train_test(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        if self.time_sensitive:
            X.sort_values("DateTime", inplace=True)
            n = len(X)
            X_train = X.iloc[:int(np.ceil(0.8 * n))]
            X_test = X.iloc[int(np.ceil(0.8 * n)):]
            y_train = y.iloc[:int(np.ceil(0.8 * n))]
            y_test = y.iloc[int(np.ceil(0.8 * n)):]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.use_dummies:
            dummied_group = ["product", "product_category", "gender", "campaign_id", "webpage_id", "user_group_id"]
            X_train = pd.get_dummies(X_train, columns=dummied_group, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=dummied_group, drop_first=True)

        return X_train, X_test, y_train, y_test

    def preprocess(self, df: pd.DataFrame) -> tuple:
        df_clean = df.drop_duplicates().copy()
        self.logger.info(f"Initial shape: {df.shape}. After removing duplicates: {df_clean.shape}")

        if "DateTime" in df_clean.columns:
            df_clean["DateTime"] = pd.to_datetime(df_clean["DateTime"], errors="coerce")

        if self.remove_outliers:
            df_clean = self.remove_outliers(df_clean)

        if self.fillna:
            df_clean = self.fill_missing_values(df_clean)

        df_clean = self.convert_categorical(df_clean)

        X = df_clean.drop(columns=["is_click"])
        y = df_clean["is_click"]

        X_train, X_test, y_train, y_test = self.split_train_test(X, y)

        return df_clean, X_train, X_test, y_train, y_test
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df["day_of_week"] = df["DateTime"].dt.dayofweek
        df["hour"] = df["DateTime"].dt.hour
        df["day"] = df["DateTime"].dt.day
        df["month"] = df["DateTime"].dt.month
        df["year"] = df["DateTime"].dt.year
        return df

    def save_data(self, df_clean, X_train, X_test, y_train, y_test):
        if self.save_as_pickle:
            df_clean.to_pickle(self.output_path / "cleaned_data.pkl")
            X_train.to_pickle(self.output_path / "X_train.pkl")
            X_test.to_pickle(self.output_path / "X_test.pkl")
            y_train.to_pickle(self.output_path / "y_train.pkl")
            y_test.to_pickle(self.output_path / "y_test.pkl")
            self.logger.info(f"Saved preprocessed data and splits as Pickle to {self.output_path}")
        else:
            df_clean.to_csv(self.output_path / "cleaned_data.csv", index=False)
            X_train.to_csv(self.output_path / "X_train.csv", index=False)
            X_test.to_csv(self.output_path / "X_test.csv", index=False)
            y_train.to_csv(self.output_path / "y_train.csv", index=False, header=True)
            y_test.to_csv(self.output_path / "y_test.csv", index=False, header=True)
            self.logger.info(f"Saved preprocessed data and splits as CSV to {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/raw/train_dataset_full.csv", help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save the output data")
    parser.add_argument("--remove-outliers", action="store_true", help="Flag to remove outliers")
    parser.add_argument("--fillna", action="store_true", help="Flag to fill missing values")
    parser.add_argument("--use-dummies", action="store_true", help="Flag to apply pd.get_dummies")
    parser.add_argument("--save-as-pickle", action="store_true", default=True, help="Flag to save as Pickle instead of CSV")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        output_path=Path(args.output_path),
        remove_outliers=args.remove_outliers,
        fillna=args.fillna,
        use_dummies=args.use_dummies,
        save_as_pickle=args.save_as_pickle
    )

    df = preprocessor.load_data(Path(args.csv_path))
    df_clean, X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test)
