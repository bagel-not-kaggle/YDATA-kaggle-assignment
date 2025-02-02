import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self):
        self.pipeline = None  # Will hold the fitted ColumnTransformer
        logger.info("Preprocessor initialized.")

    @staticmethod
    def load_data(data_path: Path) -> pd.DataFrame:
        """Load data from a CSV file."""
        logger.info(f"Loading data from {data_path}...")
        return pd.read_csv(data_path)

    def drop_nan(self, data: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
        """Drop rows with NaN values in selected columns."""
        logger.info("Dropping rows with NaN values...")
        mask = data[selected_columns].notna().all(axis=1)
        return data[mask]

    def drop_columns(self, data: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
        """Drop specified columns from the DataFrame."""
        logger.info(f"Dropping columns: {drop_columns}...")
        return data.drop(columns=drop_columns, errors='ignore')

    def fill_nan_users(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in user-related features using forward and backward fill."""
        logger.info("Filling NaN values for user-related features...")
        user_cols = ['gender', 'age_level', 'user_group_id']
        if 'user_id' in data.columns:
            filled = data.groupby('user_id')[user_cols].transform(lambda x: x.ffill().bfill())
            data[user_cols] = filled
        return data

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to extract time-based features and create new interactions."""
        logger.info("Applying feature engineering...")
        if 'DateTime' in data.columns:
            dt = pd.to_datetime(data['DateTime'])
            data['hour'] = dt.dt.hour
            data['day_of_week'] = dt.dt.dayofweek
            data['month'] = dt.dt.month
            data.drop(['DateTime'], axis=1, inplace=True)
        # Example interaction: adjust these if your data doesn't have these columns.
        if 'product_category_1' in data.columns and 'user_depth' in data.columns:
            data['product_user_depth'] = data['product_category_1'] * data['user_depth']
        return data

    def feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select features by dropping unnecessary columns."""
        logger.info("Applying feature selection...")
        drop_cols = ['session_id', 'user_id', 'campaign_id', 'webpage_id']
        return data.drop(columns=drop_cols, errors='ignore')

    def _build_and_fit_pipeline(self, X: pd.DataFrame):
        """Create and fit a ColumnTransformer on the numeric and categorical features."""
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Fitting pipeline on numeric features: {numeric_features} and categorical features: {categorical_features}.")

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        return self.pipeline.fit_transform(X)

    def preprocess_data(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Preprocess the raw data:
         - Clean data (drop NaNs, fill user-related missing values)
         - Apply feature engineering and selection
         - Fit the transformation pipeline and split into training and testing sets.
        """
        logger.info("Starting data preprocessing...")

        # Define columns that must be non-null
        required_columns = ['session_id', 'DateTime', 'user_id', 'product', 'campaign_id', 'webpage_id', 'is_click']
        data = self.drop_nan(data, required_columns)
        data = self.drop_columns(data, ['city_development_index'])
        data = self.fill_nan_users(data)
        data = self.feature_engineering(data)
        data = self.feature_selection(data)

        # Separate features and target
        if 'is_click' not in data.columns:
            logger.error("Target column 'is_click' is missing from the data!")
            raise ValueError("Missing 'is_click' column.")
        X = data.drop('is_click', axis=1)
        y = data['is_click']

        # Build and fit the transformation pipeline
        X_processed = self._build_and_fit_pipeline(X)

        logger.info("Preprocessing completed. Splitting data into train and test sets.")
        return train_test_split(X_processed, y, test_size=test_size, random_state=random_state)

    def transform(self, data: pd.DataFrame):
        """
        Transform new data using the already fitted pipeline.
        Make sure the new data goes through the same cleaning, engineering, and selection steps.
        """
        logger.info("Transforming new data for prediction...")
        data = self.fill_nan_users(data)
        if 'DateTime' in data.columns:
            data = self.feature_engineering(data)
        data = self.feature_selection(data)

        if self.pipeline is None:
            logger.error("The transformation pipeline has not been fitted yet.")
            raise ValueError("Call preprocess_data on training data before transforming new data.")

        return self.pipeline.transform(data)
