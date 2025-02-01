import pandas as pd
from pathlib import Path
from sklearn.feature_selection import chi2
# from sklearn.feature_selection import SelectKBest
import numpy as np
# import wandb

class Preprocessor:
    def __init__(self):
        self.user_labels = ['gender', 'age_level', 'user_group_id']
        self.categorical_columns =['product', 'gender','user_group_id', 'campaign_id', 'webpage_id','product_category_1']
        # self.wandb_run = wandb.init(project="shay-kaggle-competition")

    def load_data(self,data_dir: Path) -> pd.DataFrame:
        return pd.read_csv(data_dir)

    def drop_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        selected_columns = ['session_id', 'DateTime', 'user_id', 'product', 'campaign_id', 'webpage_id', 'is_click']
        mask = data[selected_columns].notna().all(axis=1)
        # self.wandb_run.log({"dropped_nan_rows": selected_columns})
        return data[mask]

    def drop_cloumns(self, data: pd.DataFrame) -> pd.DataFrame:
        # self.wandb_run.log({"dropped_nan_coulmns": drop_columns})
        drop_columns = ['session_id', 'DateTime','user_id', 'product_category_2','city_development_index']
        return data.drop(columns=drop_columns, errors='ignore')

    def drop_dupliaction(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop_duplicates()

    def feature_engineering(self, data):
        # Example: Extract date-related features
        data['hour'] = pd.to_datetime(data['DateTime']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['DateTime']).dt.dayofweek
        data['month'] = pd.to_datetime(data['DateTime']).dt.month

        # Interaction features (example)
        data['product_user_depth'] = data['product_category_1'] * data['user_depth']

        # Add user features
        data = self.create_user_features(data)
        print(f"create_user_features: {data.shape}")

        # Add campaign features
        data = self.create_campaign_features(data)
        print(f"create_campaign_features: {data.shape}")

        # Add product features
        data = self.create_product_features(data)
        print(f"create_product_features: {data.shape}")

        #drop features
        data = self.drop_cloumns(data)
        print(f"drop_cloumns: {data.shape}")

        # self.wandb_run.log({})
        return data

    def get_dummies(self, data):
        for col in self.categorical_columns:
            data[col] = data[col].astype(str)

        encoded_categorical = pd.get_dummies(data[self.categorical_columns],
                                             columns=self.categorical_columns,
                                             drop_first=True,
                                             dummy_na=False)

        # 3. Concatenate encoded categorical features with numerical features:
        data = data.drop(self.categorical_columns, axis=1)
        data = pd.concat([data, encoded_categorical], axis=1)
        return data

    # Function to fill NaN values based on user mapping
    def fill_user_info(self, row):
        user_id = row['user_id']
        for label in self.user_labels:
            if pd.isna(row[label]):
                row[label] = self.user_info_mapping.loc[
                    user_id, label] if user_id in self.user_info_mapping.index and not pd.isna(
                    self.user_info_mapping.loc[user_id, label]) else row[label]
        return row

    def remove_highly_correlated_features(self, data, threshold=0.1):
      # Calculate correlation matrix
      numerical_columns = []
      numerical_columns = data[numerical_columns].select_dtypes(include=['number']).columns
      data = data[numerical_columns]
      correlation_matrix = data.corr()

      # Select features highly correlated with the target
      correlated_features = correlation_matrix['is_click'][abs(correlation_matrix['is_click']) > threshold].index
      return correlated_features

    def select_features_chi2(self, data, alpha=0.05):
        categorical_features = ['user_group_id', 'campaign_id', 'webpage_id', 'product_category_1','is_click']
        data = data.dropna(subset=categorical_features)
        X = data[categorical_features]
        y = data['is_click']

        # Perform Chi-Square Test
        chi_scores, p_values = chi2(X, y)

        results = pd.DataFrame({
            'Feature': X.columns,
            'Chi-Square Score': chi_scores,
            'p-value': p_values
        })

        # Select features with p-value < 0.05
        return results[results['p-value'] < alpha]

    def create_user_features(self, train_dataset):

        # User-Specific CTR
        user_ctr = train_dataset.groupby('user_id')['is_click'].mean().rename('user_ctr')

        # User-Campaign Interaction
        user_campaign_ctr = train_dataset.groupby(['user_id', 'campaign_id'])['is_click'].mean().rename(
            'campaign_ctr').reset_index()
        user_campaign_clicks = train_dataset.groupby(['user_id', 'campaign_id'])['is_click'].sum().rename(
            'campaign_clicks').reset_index()

        # Frequency-based features
        user_product_views = train_dataset.groupby(['user_id', 'product'])['session_id'].count().rename(
            'user_product_views').reset_index()
        user_category_views = train_dataset.groupby(['user_id', 'product_category_1'])['session_id'].count().rename(
            'user_category_views').reset_index()

        # Aggregated category-level features
        category_ctr = train_dataset.groupby('product_category_1')['is_click'].mean().rename(
            'category_ctr').reset_index()

        # Total Number of Sessions
        user_sessions = train_dataset.groupby('user_id')['session_id'].nunique().rename('total_sessions').reset_index()

        # Merge features
        user_features = pd.merge(train_dataset, user_ctr, on='user_id', how='left')
        user_features = pd.merge(user_features, user_campaign_ctr, on=['user_id', 'campaign_id'], how='left')
        user_features = pd.merge(user_features, user_campaign_clicks, on=['user_id', 'campaign_id'], how='left')
        user_features = pd.merge(user_features, user_product_views, on=['user_id', 'product'], how='left')
        user_features = pd.merge(user_features, user_category_views, on=['user_id', 'product_category_1'], how='left')
        user_features = pd.merge(user_features, category_ctr, on=['product_category_1'], how='left')
        user_features = pd.merge(user_features, user_sessions, on=['user_id'], how='left')

        user_features.fillna(0, inplace=True)  # Fill NaN (for users with no views) with 0

        return user_features

    def create_campaign_features(self, train_dataset):

        campaign_ctr = train_dataset.groupby('campaign_id')['is_click'].mean().rename('campaign_ctr')
        campaign_clicks = train_dataset.groupby('campaign_id')['is_click'].sum().rename('campaign_clicks')
        campaign_views = train_dataset.groupby('campaign_id')['session_id'].count().rename('campaign_views')

        campaign_features = pd.concat([campaign_ctr, campaign_clicks, campaign_views], axis=1).reset_index()

        # Campaign-Product Interaction
        campaign_product_ctr = train_dataset.groupby(['campaign_id', 'product'])['is_click'].mean().rename(
            'product_ctr').reset_index()
        campaign_product_clicks = train_dataset.groupby(['campaign_id', 'product'])['is_click'].sum().rename(
            'product_clicks').reset_index()

        # Frequency-based features
        campaign_product_views = train_dataset.groupby(['campaign_id', 'product'])['session_id'].count().rename(
            'campaign_product_views').reset_index()
        campaign_category_views = train_dataset.groupby(['campaign_id', 'product_category_1'])['session_id'].count().rename(
            'campaign_category_views').reset_index()

        # Merge features
        campaign_features = pd.merge(train_dataset, campaign_features, on='campaign_id', how='left')
        campaign_features = pd.merge(campaign_features, campaign_product_ctr, on=['campaign_id', 'product'], how='left')
        campaign_features = pd.merge(campaign_features, campaign_product_clicks, on=['campaign_id', 'product'], how='left')
        campaign_features = pd.merge(campaign_features, campaign_product_views, on=['campaign_id', 'product'], how='left')
        campaign_features = pd.merge(campaign_features, campaign_category_views, on=['campaign_id', 'product_category_1'], how='left')

        campaign_features.fillna(0, inplace=True)

        return campaign_features

    def create_product_features(self, train_dataset):

            product_ctr = train_dataset.groupby('product')['is_click'].mean().rename('product_ctr')
            product_clicks = train_dataset.groupby('product')['is_click'].sum().rename('product_clicks')
            product_views = train_dataset.groupby('product')['session_id'].count().rename('product_views')

            product_features = pd.concat([product_ctr, product_clicks, product_views], axis=1).reset_index()

            # Product-Category Interaction
            product_category_ctr = train_dataset.groupby(['product', 'product_category_1'])['is_click'].mean().rename(
                'category_ctr').reset_index()
            product_category_clicks = train_dataset.groupby(['product', 'product_category_1'])['is_click'].sum().rename(
                'category_clicks').reset_index()

            # Frequency-based features
            product_category_views = train_dataset.groupby(['product', 'product_category_1'])['session_id'].count().rename(
                'product_category_views').reset_index()
            product_user_views = train_dataset.groupby(['product', 'user_id'])['session_id'].count().rename(
                'product_user_views').reset_index()

            # Merge features
            product_features = pd.merge(train_dataset, product_features, on='product', how='left')
            product_features = pd.merge(product_features, product_category_ctr, on=['product', 'product_category_1'], how='left')
            product_features = pd.merge(product_features, product_category_clicks, on=['product', 'product_category_1'], how='left')
            product_features = pd.merge(product_features, product_category_views, on=['product', 'product_category_1'], how='left')
            product_features = pd.merge(product_features, product_user_views, on=['product', 'user_id'], how='left')

            product_features.fillna(0, inplace=True)

            return product_features

    def preprocess_data(self, data):
        # drop nan values
        data = self.drop_nan(data)

        data = self.drop_dupliaction(data)

        ### fill user nan values
        # 1. Identify users with at least one NaN in the specified columns
        users_with_nan = data[data[self.user_labels].isna().any(axis=1)]['user_id'].unique()

        # 2. Filter the dataset to include only these users
        data_with_nan = data[data['user_id'].isin(users_with_nan)]

        # 3. Group by 'user_id' and get the most frequent value for each label (only for users with NaN)
        self.user_info_mapping = data.groupby('user_id')[self.user_labels].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan)

        # 4. Apply the function only to the filtered dataset with NaN values
        filtered_dataset_with_nan = data_with_nan.apply(self.fill_user_info, axis=1)

        # 5. Update the original dataset with the imputed values
        data.update(filtered_dataset_with_nan)  # This merges the imputed values back into the original dataset

        # Apply feature engineering
        data = self.feature_engineering(data)

        # Apply Dummies
        data = self.get_dummies(data)

        return data



if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Load the data
    data_path = 'data/raw/train_dataset_full.csv'
    preprocessor = Preprocessor()
    data = preprocessor.load_data(data_path)

    data = preprocessor.preprocess_data(data)

    # Separate features and target
    X = data.drop('is_click', axis=1)
    y = data['is_click']
    # Initialize the RandomForestClassifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    # Calculate evaluation metrics
    f1_score = f1_score(y, y_pred)
    confusion_matrix = confusion_matrix(y, y_pred)
    classification_report = classification_report(y, y_pred)

    print(f"F1 Score: {f1_score:.2f}")
    print("Classification Report:")
    print(classification_report)

    print("Confusion Matrix:")
    # Display the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()