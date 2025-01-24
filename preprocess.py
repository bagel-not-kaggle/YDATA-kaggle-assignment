# Preprocess

import pandas as pd
from pathlib import Path

class Preprocessor:
    def __init__(self):
        # self.scaler = StandardScaler()

    def load_data(self,data_dir: Path) -> pd.DataFrame:
        return pd.read_csv(data_dir)

    def drop_nan(self, data: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
        mask = data[selected_columns].notna().all(axis=1)
        return data[mask]

    def drop_cloumns(self, data: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
        return data.drop(columns=drop_columns, errors='ignore')

    def fill_nan_users(self, data: pd.DataFrame) -> pd.DataFrame:
        labels = ['gender', 'age_level', 'user_group_id']
        pass

    def feature_engineering(self, data):
        # Example: Extract date-related features
        data['hour'] = pd.to_datetime(data['DateTime']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['DateTime']).dt.dayofweek
        data['month'] = pd.to_datetime(data['DateTime']).dt.month

        # Drop original DateTime column
        data.drop(['DateTime'], axis=1, inplace=True)

        # Interaction features (example)
        data['product_user_depth'] = data['product_category_1'] * data['user_depth']

        return data

    def feature_selection(self, data):
        # Drop irrelevant or highly correlated columns
        drop_cols = ['session_id', 'user_id', 'campaign_id', 'webpage_id']
        data = data.drop(columns=drop_cols, errors='ignore')

        return data

    def get_dummies(self, data):
        columns = ['product', 'gender']
        encoded_categorical = pd.get_dummies(data[columns],
                                             columns=columns,
                                             drop_first=True,
                                             dummy_na=False)

        # 3. Concatenate encoded categorical features with numerical features:
        data = data.drop(columns, axis=1)
        data = pd.concat([data, encoded_categorical], axis=1)
        return data


    def preprocess_data(self, data):
        # drop nan values
        selected_columns = ['session_id', 'DateTime', 'user_id', 'product', 'campaign_id', 'webpage_id', 'is_click']
        data = self.drop_nan(data, selected_columns)

        # Drop columns with high percentage of missing values
        drop_columns = ['city_development_index']
        data = self.drop_cloumns(data, drop_columns)

        #fill user nan values
        # data = self.fill_nan_users(data)

        # Apply feature engineering
        data = self.feature_engineering(data)

        # Apply feature selection
        data = self.feature_selection(data)

        # Apply Dummies
        data = self.feature_selection(data)

        return data