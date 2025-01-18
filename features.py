import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from preprocessed data
    """
    # Create features
    X = pd.get_dummies(df, columns=['product', 'gender'], drop_first=True)

    # Drop non-feature columns
    X = X.drop(['DateTime', 'user_id', 'session_id'], axis=1)

    return X


def get_target(df: pd.DataFrame) -> pd.Series:
    """
    Extract target variable
    """
    return df['is_click']