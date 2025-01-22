import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from pathlib import Path

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    def load_data(data_dir: Path) -> pd.DataFrame:
        return pd.read_csv(data_dir)

    def drop_nan(data: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
        mask = data[selected_columns].notna().all(axis=1)
        return data[mask]

    def drop_cloumns(data: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
        return data.drop(columns=drop_columns, errors='ignore')

    def fill_nan_users(data: pd.DataFrame) -> pd.DataFrame:
        labels = ['gender', 'age_level', 'user_group_id']
        filled_features = (data.groupby('user_id')[labels]
                           .transform(lambda x: x.ffill().bfill())
                           .infer_objects(copy=False))  # Add infer_objects
        data[labels] = filled_features

    def feature_engineering(self, data):
        # Example: Extract date-related features
        data['hour'] = pd.to_datetime(data['datetime']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['datetime']).dt.dayofweek
        data['month'] = pd.to_datetime(data['datetime']).dt.month

        # Drop original datetime column
        data.drop(['datetime'], axis=1, inplace=True)

        # Interaction features (example)
        data['product_user_depth'] = data['product_category_1'] * data['user_depth']

        return data

    def feature_selection(self, data):
        # Drop irrelevant or highly correlated columns
        drop_cols = ['session_id', 'user_id', 'campaign_id', 'webpage_id']
        data = data.drop(columns=drop_cols, errors='ignore')

        return data

    def preprocess_data(self, data):
        # drop nan values
        selected_columns = ['session_id', 'DateTime', 'user_id', 'product', 'campaign_id', 'webpage_id', 'is_click']
        data = self.drop_nan(data, selected_columns)

        # Drop columns with high percentage of missing values
        drop_columns = ['city_development_index']
        data = self.drop_cloumns(data, drop_columns)

        #fill user nan values
        data = self.fill_nan_users(data)

        # Apply feature engineering
        data = self.feature_engineering(data)

        # Apply feature selection
        data = self.feature_selection(data)

        # Separate features and target
        X = data.drop('is_click', axis=1)
        y = data['is_click']

        # Define numerical and categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.scaler, numeric_features),
                ('cat', self.ohe, categorical_features)
            ]
        )

        X_processed = preprocessor.fit_transform(X)
        return train_test_split(X_processed, y, test_size=0.2, random_state=42)