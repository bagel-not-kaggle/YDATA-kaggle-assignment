import argparse
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

class DataPreprocessor:
    def __init__(
        self,
        output_path: Path = Path("data/processed"),
        remove_outliers: bool = False,
        fillna: bool = False,
        use_dummies: bool = True,
        save_as_pickle: bool = True,
        callback=None
    ):
        self.output_path = output_path
        self.remove_outliers = remove_outliers
        self.fillna = fillna
        self.use_dummies = use_dummies
        self.save_as_pickle = save_as_pickle
        self.callback = callback or (lambda x: None)  # Default no-op callback if none provided
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)
        self.callback({
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
        "initial_missing_values": df.isna().sum().sum()
        })
        self.logger.info(f"Loading file from: {csv_path}")
        return df

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
            self.callback({
        "rows_after_outlier_removal": len(df),
        "outliers_removed": outlier_condition.sum()
    })
        return df

    def fill_missing_values(self, df: pd.DataFrame, use_mode: bool = False) -> pd.DataFrame:
        """
        Fill missing values using mode, median, or forward/backward fill.
        Includes subfunctions for modularity.
        """
        df = df.copy()
        df["user_id"] = df["user_id"].fillna(-1).astype("int32")
        # Subfunction for filling with mode
        def _fill_with_mode(df, columns):
            for column in columns:
                if column in df.columns:
                    mode_value = df[column].mode()
                    if not mode_value.empty:
                        df[column] = df[column].fillna(mode_value.iloc[0])
            return df

        # Subfunction for filling with median
        def _fill_with_median(df, columns):
            for column in columns:
                if column in df.columns:
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
            return df

        # Subfunction for forward/backward filling (requires sorting)
        def _fill_with_ffill_bfill_user(df, columns):
            if "DateTime" in df.columns:
                df = df.sort_values("DateTime")
            df[columns] = (
            df.groupby("user_id",observed = True)[columns]
            .transform(lambda x: x.ffill().bfill())
            .infer_objects(copy=False)
        )
            if "DateTime" in df.columns:
                df = df.sample(frac=1)  # Shuffle rows back to avoid keeping sort order
            return df
        
        
        df["product_category"] = df["product_category_1"].fillna(df["product_category_2"])
        df.drop(columns=["product_category_1", "product_category_2"], inplace=True)

        # Define columns to fill
        columns_to_fill_mode = ["product", "campaign_id", "webpage_id", "gender", "var_1", "product_category"]

        # Apply mode-based filling if enabled
        if use_mode:
            df = _fill_with_mode(df, columns_to_fill_mode)

        # Apply median-based filling
        #self.logger.info("Filled missing values with median.")

        # Apply forward/backward filling
        cols_for_ffill_bfill1 = ["product", "campaign_id", "webpage_id", "gender",  "product_category"]
        cols_for_ffill_bfill2 = ["age_level", "city_development_index","var_1", "user_depth"]
        self.logger.info(f"Filling ffil bfil missing values for columns: {cols_for_ffill_bfill2}")
        df = _fill_with_ffill_bfill_user(df, cols_for_ffill_bfill2)
        self.logger.info("Filled missing values with forward/backward fill.")
        missing_is_click = df["is_click"].isna().sum()
        if missing_is_click > 0:
            self.logger.warning(f"{missing_is_click} rows still have missing 'is_click'. Dropping these rows.")
            df = df.dropna(subset=["is_click"])

        
        return df

    def determine_categorical_features(self, df: pd.DataFrame, cat_features: list = None):
        """
        Identify and process categorical features, ensuring compatibility with CatBoost.
        """
        cat_cols = ["product_category", "product", "gender", "campaign_id", "webpage_id", "user_group_id"]
        for col in cat_cols:
            if col in df.columns:
                if col in ["campaign_id", "webpage_id", "user_group_id", "product_category"]:
                    df[col] = df[col].astype("Int64").astype("category")
                else:
                    df[col] = df[col].astype("category")
        if cat_features:
            cat_features = [col for col in cat_features if col in df.columns]
        else:
            cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in cat_features:
            if col in df.columns:

                # Add "missing" only if it's not already a category
                if "missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("missing")

                # Fill missing values with "missing"
                df[col] = df[col].fillna("missing")

        return cat_features

    def feature_generation2(self, df: pd.DataFrame, use_missing_with_mode=False, get_dumm=False, catb=True):
        df = df.copy()

        # Generate time-based features
        df['Day'] = df['DateTime'].dt.day
        df['Hour'] = df['DateTime'].dt.hour
        df['Minute'] = df['DateTime'].dt.minute
        df['weekday'] = df['DateTime'].dt.weekday

        cols_to_fill = ["Day", "Hour", "Minute", "weekday"]   
        self.logger.info(f"Filling missing values for columns: {cols_to_fill}")
        df[cols_to_fill] = (df.groupby("user_id",observed = True)[cols_to_fill]
            .transform(lambda x: x.ffill().bfill())
            .infer_objects(copy=False)
        )
        colls_to_fill_nas = df[cols_to_fill].isna().sum()
        if colls_to_fill_nas.sum() > 0:
            self.logger.warning(f"Still missing values in columns: {colls_to_fill_nas.sum()}")
            df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].mode().iloc[0])
        self.logger.info("Filled missing values with forward/backward fill.")
        # Fill missing values
        if use_missing_with_mode:
            df = self.fill_missing_values(df, use_mode=True)

        # Generate campaign-based features
        df['start_date'] = df.groupby('campaign_id', observed=True)['DateTime'].transform('min')
        df['campaign_duration'] = df['DateTime'] - df['start_date']
        df['campaign_duration_days'] = df['campaign_duration'].dt.total_seconds() / (3600)
        df['campaign_duration_days'] = df['campaign_duration_days'].fillna(
            df.groupby('campaign_id', observed=True)['campaign_duration_days'].transform(lambda x: x.mode().iloc[0]))
        df['campaign_duration_days'] = df.groupby('webpage_id', observed=True)['campaign_duration_days'].transform(
            lambda x: x.ffill().bfill() if not x.mode().empty else x.fillna(0))

        # Drop unnecessary columns
        df.drop(columns=['DateTime', 'start_date', 'campaign_duration', 'session_id', 'user_id', 'user_group_id'], inplace=True)
      
        if catb:
            self.determine_categorical_features(df)
        # One-hot encoding if `get_dumm` is True
        if get_dumm:
            columns_to_d = ["product", "campaign_id", "webpage_id", "product_category", "gender"]
            df = pd.get_dummies(df, columns=columns_to_d)

        self.callback({
        "total_features": len(df.columns),
        "categorical_features": len(df.select_dtypes(include=['object', 'category']).columns),
        "numerical_features": len(df.select_dtypes(include=['int64', 'float64']).columns)
            })

        return df

    def preprocess(self, df: pd.DataFrame) -> tuple:
        df_clean = df.drop_duplicates().copy()
        self.logger.info(f"Initial shape: {df.shape}. After removing duplicates: {df_clean.shape}")

        if "DateTime" in df_clean.columns:
            df_clean["DateTime"] = pd.to_datetime(df_clean["DateTime"], errors="coerce")

        if self.remove_outliers:
            df_clean = self.remove_outliers(df_clean)

        if self.fillna:
            df_clean = self.fill_missing_values(df_clean)

        df_clean = self.feature_generation2(df_clean)

        X = df_clean.drop(columns=["is_click"])
        y = df_clean["is_click"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create stratified folds for train set
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_datasets = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
            fold_datasets.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
            self.callback({
            f"fold_{fold}_train_size": len(train_idx),
            f"fold_{fold}_val_size": len(val_idx),
            f"fold_{fold}_positive_ratio": y_train.iloc[train_idx].mean()
            })

        self.logger.info("Created stratified folds for training data.")

        return df_clean, X_train, X_test, y_train, y_test, fold_datasets

    def save_data(self, df_clean, X_train, X_test, y_train, y_test, fold_datasets):
        if self.save_as_pickle:
            df_clean.to_pickle(self.output_path / "cleaned_data.pkl")
            X_train.to_pickle(self.output_path / "X_train.pkl")
            X_test.to_pickle(self.output_path / "X_test.pkl")
            y_train.to_pickle(self.output_path / "y_train.pkl")
            y_test.to_pickle(self.output_path / "y_test.pkl")
            
            # Save folds
            for i, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(fold_datasets):
                X_train_fold.to_pickle(self.output_path / f"X_train_fold_{i}.pkl")
                y_train_fold.to_pickle(self.output_path / f"y_train_fold_{i}.pkl")
                X_val_fold.to_pickle(self.output_path / f"X_val_fold_{i}.pkl")
                y_val_fold.to_pickle(self.output_path / f"y_val_fold_{i}.pkl")

            self.logger.info(f"Saved preprocessed data, train-test split, and folds as Pickle to {self.output_path}")
        else:
            df_clean.to_csv(self.output_path / "cleaned_data.csv", index=False)
            X_train.to_csv(self.output_path / "X_train.csv", index=False)
            X_test.to_csv(self.output_path / "X_test.csv", index=False)
            y_train.to_csv(self.output_path / "y_train.csv", index=False, header=True)
            y_test.to_csv(self.output_path / "y_test.csv", index=False, header=True)
            
            # Save folds
            for i, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(fold_datasets):
                X_train_fold.to_csv(self.output_path / f"X_train_fold_{i}.csv", index=False)
                y_train_fold.to_csv(self.output_path / f"y_train_fold_{i}.csv", index=False, header=True)
                X_val_fold.to_csv(self.output_path / f"X_val_fold_{i}.csv", index=False)
                y_val_fold.to_csv(self.output_path / f"y_val_fold_{i}.csv", index=False, header=True)

            self.logger.info(f"Saved preprocessed data, train-test split, and folds as CSV to {self.output_path}")

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
    df_clean, X_train, X_test, y_train, y_test, fold_datasets = preprocessor.preprocess(df)
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test, fold_datasets)
