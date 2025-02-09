import argparse
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import numpy as np
from typing import Tuple

class DataPreprocessor:
    def __init__(
        self,
        output_path: Path = Path("data/processed"),
        remove_outliers: bool = False,
        fillna: bool = False,
        use_dummies: bool = False,
        save_as_pickle: bool = True,
        catb: bool = True,
        use_missing_with_mode: bool = False,
        fill_cat: bool = False,

        callback=None
    ):
        self.output_path = output_path
        self.remove_outliers = remove_outliers
        self.catb = catb
        self.fillna = fillna
        self.use_dummies = use_dummies
        self.use_missing_with_mode = use_missing_with_mode
        self.save_as_pickle = save_as_pickle
        self.callback = callback or (lambda x: None)  # Default no-op callback if none provided
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.fill_cat = fill_cat
        

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    
    
    """       
    ╦ ╦┌─┐┬  ┌─┐┌─┐┬─┐  ╔═╗┬ ┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌┌─┐
    ╠═╣├┤ │  ├─┘├┤ ├┬┘  ╠╣ │ │││││   │ ││ ││││└─┐
    ╩ ╩└─┘┴─┘┴  └─┘┴└─  ╚  └─┘┘└┘└─┘ ┴ ┴└─┘┘└┘└─┘

    """
    def load_all_data(self, train_csv_path: Path, test_csv_path: Path= None) -> pd.DataFrame:
        if not train_csv_path.exists():
            raise FileNotFoundError(f"File not found: {train_csv_path}")
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
               
        self.logger.info(f"Loading train from: {train_csv_path} and test from {test_csv_path}")
        return df_train, df_test


    def load_data(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)
 
        self.logger.info(f"Loading file from: {csv_path}")
        return df
    
    def drop_completely_empty(self, df: pd.DataFrame) -> pd.DataFrame:
        '''drop completely empty rows'''
        df.dropna(how='all', inplace=True)
        return df
    
    def drop_na_session_id_or_is_click(self, df: pd.DataFrame) -> pd.DataFrame:
        '''drop rows missing session_id or is_click'''
        df.dropna(subset=["session_id","is_click"], inplace=True)
        return df
    
    def drop_dup_session_id(self, df: pd.DataFrame) -> pd.DataFrame:
        '''drop rows with identical session_ids'''
        df.drop_duplicates(subset=["session_id"], inplace=True)
        return df

    def decrease_test_user_group_id(self, df_test: pd.DataFrame) -> pd.DataFrame:
        '''decrease user_group_id by 1 to align with training data'''
        df_test["user_group_id"] = df_test["user_group_id"] - 1
        return df_test

    def replace_test_user_depth_to_training(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        '''replace (or fill) user_depth values in X_test according to the first
         valid user_depth observed per user in X_train.'''
        depth_mapping = (
        df_train
        .dropna(subset=["user_depth"])           
        .groupby("user_id")["user_depth"]        
        .first()                                 
        .to_dict()                               
        )

        df_test["user_depth"] = (
        df_test["user_id"].map(depth_mapping)
        .fillna(df_test["user_depth"])
        )
        return df_test

    def concat_train_test(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        '''concat df_train with df_test, with is_click = -1 as an indicator'''
        df_test["is_click"] = np.full(df_test.shape[0], -1)
        df = pd.concat([df_train, df_test], ignore_index=True)
        return df
    
    # Function to infer missing values based on user_id
    def infer_by_col(self, df: pd.DataFrame, target_col, key_col='user_id', mapping_df=None)-> pd.DataFrame:
        """
        Infers missing values in `target_col` based on a mapping from `key_col`.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            target_col (str): The column to fill missing values in.
            key_col (str): The column used as the key for mapping (default is 'user_id').
        
        Returns:
            pd.DataFrame: The DataFrame with missing values in `target_col` filled.
        """
        # Use the entire DataFrame or the provided mapping DataFrame
        if mapping_df is not None:
            source_df = mapping_df
        else:
            source_df = df

        # Create a dictionary mapping key_col to target_col, ignoring NaNs
        mapping_dict = (
            source_df.dropna(subset=[target_col])
            .groupby(key_col)[target_col]
            .first()  # Assumes there's only one unique value per key_col
            .to_dict()
        )
        
        # Map the key_col to target_col
        mapped_values = df[key_col].map(mapping_dict)
        
        # Fill missing values in target_col
        df[target_col] = df[target_col].fillna(mapped_values)
        
        return df

    # Function to infer missing values based on group combinations
    def infer_by_two_cols(self, df: pd.DataFrame, target_col, key_cols, mapping_df=None)-> pd.DataFrame:
        """
        Infers missing values in `target_col` based on a unique mapping from `group_cols`.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            group_cols (list of str): The columns to group by for mapping.
            target_col (str): The column to fill missing values in.
            mapping_df (pd.DataFrame, optional): A pre-filtered DataFrame to build the mapping from.
                                                If None, uses the entire DataFrame.
        
        Returns:
            pd.DataFrame: The DataFrame with missing values in `target_col` filled.
        """
        # Use the entire DataFrame or the provided mapping DataFrame
        if mapping_df is not None:
            source_df = mapping_df
        else:
            source_df = df
        
        # Create a dictionary mapping group_cols to target_col, ignoring NaNs
        mapping_dict = (
            source_df.dropna(subset=[target_col])
                    .groupby(key_cols)[target_col]
                    .first()  # Assumes there's only one unique value per group
                    .to_dict()
        )
        
        # Create a MultiIndex based on group_cols and map to target_col
        mapped_values = df.set_index(key_cols).index.map(mapping_dict)
        
        # Convert the mapped values to a Series aligned with the original DataFrame
        mapped_series = pd.Series(mapped_values, index=df.index)
        
        # Fill missing values in target_col
        df[target_col] = df[target_col].fillna(mapped_series)
        
        return df
    
    def fillna_when_single_unique_value(self, df: pd.DataFrame, group_col)-> pd.DataFrame:
        """
        For each group (based on group_col) and for each column:
        - if that column has exactly one unique non-null value within the group,
            fill any NaNs in that column with the single unique value.
        """
        df = df.copy()  # avoid mutating original
        
        fillable_cols = ['DateTime', 'user_id', 'product', 'campaign_id',
       'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'age_level', 'user_depth',
       'city_development_index', 'var_1']
        
        for col in fillable_cols:
            # For each group, collect the array of unique non-null values
            group_unique = (
                df.groupby(group_col)[col]
                .apply(lambda x: x.dropna().unique())
            )
            # Convert that array into a single value if length == 1, else np.nan
            single_val_map = group_unique.apply(
                lambda arr: arr[0] if len(arr) == 1 else np.nan
            ).to_dict()

            # Now fill the missing values for groups that have a single unique value
            def _fill_func(row):
                if pd.isnull(row[col]):
                    # Look up the single possible value for this group
                    possible_val = single_val_map.get(row[group_col], np.nan)
                    if not pd.isnull(possible_val):
                        return possible_val
                return row[col]
            
            df[col] = df.apply(_fill_func, axis=1)

        return df

    def deterministic_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        changed = True
        max_iterations = 10
        iteration = 0

        while changed and iteration < max_iterations:
            old_df = df.copy()
            user_cols = ['user_group_id', 'gender', 'age_level', 'city_development_index', 'user_depth']

            for col in user_cols:
                df = self.infer_by_col(df, col, key_col='user_id')

            #infer webpage_id from campaign_id
            df = self.infer_by_col(df, "webpage_id", key_col='campaign_id')

            df = self.infer_by_col(df, "product_category_1", key_col='campaign_id', mapping_df= df[df.campaign_id == 396664])

            df = self.infer_by_col(df, "campaign_id", key_col='webpage_id', mapping_df= df[df.webpage_id != 13787])

            df = self.infer_by_col(df, "product_category_1", key_col='webpage_id', mapping_df= df[df.webpage_id == 51181])

            df = self.infer_by_col(df, "gender", key_col='user_group_id', mapping_df= df[df.user_group_id != 0])

            df = self.infer_by_col(df, "age_level", key_col='user_group_id')

            df = self.infer_by_col(df, "user_group_id", key_col='age_level', mapping_df= df[df.age_level == 0])

            df = self.infer_by_two_cols(df, target_col="user_group_id", key_cols=["age_level", "gender"])
 
            df = self.fillna_when_single_unique_value(df, group_col= "product_category_2")

            changed = not df.equals(old_df)
            iteration += 1

        return df
    
    def split_to_train_test(self, df: pd.DataFrame) -> pd.DataFrame:
        train_df = df[df.is_click !=-1]
        test_df = df[df.is_click ==-1]
        return train_df, test_df

    def add_target_encoding(self, df_train: pd.DataFrame, df_test: pd.DataFrame, cols_to_target_encode) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Ensure we're working with copies
        df_train = df_train.copy()
        df_test = df_test.copy()
        
        # Extract the columns to encode from both train and test dataframes
        df_train_te = df_train[cols_to_target_encode].copy()
        df_test_te = df_test[cols_to_target_encode].copy()

        # Create and fit the TargetEncoder on the training set
        te = TargetEncoder(categories='auto', target_type='binary',
                        smooth='auto', cv=5, shuffle=True, random_state=42)
        
        df_train_te = te.fit_transform(df_train_te, df_train["is_click"])
        df_test_te = te.transform(df_test_te)
            
        # Append the encoded features back to the original dataframes
        for i, orig_col in enumerate(cols_to_target_encode):
            df_train.loc[:, f"{orig_col}_te"] = df_train_te[:, i]
            df_test.loc[:, f"{orig_col}_te"] = df_test_te[:, i]

        return df_train, df_test

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
    
    def fill_with_ffill_bfill_user(self,df, columns):
        if "DateTime" in df.columns:
            df = df.sort_values("DateTime")
        df[columns] = (
        df.groupby("user_id",observed = True)[columns]
        .transform(lambda x: x.ffill().bfill())
        .infer_objects(copy=False))
        if "DateTime" in df.columns:
            df = df.sample(frac=1)  # Shuffle rows back to avoid keeping sort order
        return df
    
    def _fill_with_mode(self,df, columns): # Maybe groupby user_id
        for column in columns:
            if column in df.columns:
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value.iloc[0])
        return df

    def _fill_with_median(self, df, columns): # Maybe groupby user_id
        for column in columns:
            if column in df.columns:
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
        return df

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using mode, median, or forward/backward fill.
        Includes subfunctions for modularity.
        """
        df = df.copy()
        self.logger.info(f"NAs in the dataset: {df.isna().sum().sum()}")
        df["user_id"] = df["user_id"].fillna(-1).astype("int32")        
        df["product_category"] = df["product_category_1"].fillna(df["product_category_2"])
        df.drop(columns=["product_category_1", "product_category_2"], inplace=True)

        # Define columns to fill
        cat_cols_to_fill = ["product", "campaign_id", "webpage_id", "gender", "product_category"]

        # Apply mode-based filling if enabled
        if self.fill_cat:
            df = self.fill_with_ffill_bfill_user(df, cat_cols_to_fill)

        cols_for_ffill_bfill = ["age_level", "city_development_index","var_1", "user_depth"]
        self.logger.info(f"Filling ffil bfil missing values for columns: {cols_for_ffill_bfill}")
        self.logger.info(f'Number of missing values before: {df[cols_for_ffill_bfill].isna().sum()}')
        df = self.fill_with_ffill_bfill_user(df, cols_for_ffill_bfill)
        self.logger.info(f'Number of missing values after: {df[cols_for_ffill_bfill].isna().sum()}')
        if df[cols_for_ffill_bfill].isna().sum().sum() > 0:
            self.logger.warning(f'Still missing values in columns: {cols_for_ffill_bfill}')
            df[cols_for_ffill_bfill] = df[cols_for_ffill_bfill].fillna(df[cols_for_ffill_bfill].mode().iloc[0])
        missing_is_click = df["is_click"].isna().sum()
        if missing_is_click > 0:
            self.logger.warning(f"{missing_is_click} rows still have missing 'is_click'. Dropping these rows.")
            df = df.dropna(subset=["is_click"])

        return df

    def determine_categorical_features(self, df: pd.DataFrame, cat_features: list = None): # For catboost
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
    
    """
    ╔═╗┌─┐┌─┐┌┬┐┬ ┬┬─┐┌─┐  ╔═╗┌─┐┌┐┌┌─┐┬─┐┌─┐┌┬┐┬┌─┐┌┐┌
    ╠╣ ├┤ ├─┤ │ │ │├┬┘├┤   ║ ╦├┤ │││├┤ ├┬┘├─┤ │ ││ ││││
    ╚  └─┘┴ ┴ ┴ └─┘┴└─└─┘  ╚═╝└─┘┘└┘└─┘┴└─┴ ┴ ┴ ┴└─┘┘└┘

    """
    def feature_generation(self, df: pd.DataFrame):
        df = df.copy()

        df['day_of_week'] = df.DateTime.dt.dayofweek
        df['hour'] = df.DateTime.dt.hour

        # binning hour to part_of_day
        def part_of_day(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 22:
                return 'evening'
            else:
                return 'night'

        df['part_of_day'] = df.hour.apply(part_of_day)

        # hours_since_campaign_start
        first_time_campaign_map = df.groupby("campaign_id")["DateTime"].min().to_dict()
        df['hours_since_campaign_start'] = (df.DateTime - df.campaign_id.map(first_time_campaign_map)).dt.total_seconds() // 3600

        # user_total_num_of_sessions
        df['user_total_num_of_sessions']= df.groupby("user_id")["session_id"].transform('nunique')

        # user_session_order (starting from 1)
        df['user_session_order'] = df.sort_values(by="DateTime").groupby('user_id')['DateTime'].rank(method='first')

        # is_first_session (binary)
        first_session_map = df.groupby("user_id")["DateTime"].min().to_dict()
        df["is_first_session"] = (df.DateTime == df.user_id.map(first_session_map)).astype("int")

        # user's total clicks before the current session (without leakage)
        df['user_cum_clicks'] = df.sort_values(by="DateTime").groupby('user_id')['is_click'].transform(lambda x: x.cumsum().shift(fill_value=0))

        # user_num_of_days_in_webpage
        df['user_num_of_days_in_webpage'] = df.groupby(["user_id", "webpage_id"])["day_of_week"].transform('nunique')

        # campaign_num_of_products
        df['campaign_num_of_products'] = df.groupby("campaign_id")["product"].transform('nunique')

        # campaign_num_of_product_categories
        df['campaign_num_of_product_categories'] = df.groupby("campaign_id")["product_category_1"].transform('nunique')

        # user's hours_since_last_session
        df['hours_since_last_session'] = (df.sort_values(by="DateTime").groupby("user_id")["DateTime"].diff()).dt.total_seconds()/3600

        # user's past CTR (without leakage)
        df['user_past_ctr'] = df.sort_values('DateTime').groupby('user_id')['is_click'].transform(lambda s: s.expanding().mean().shift())

        # user's hours hours_since_last_click (without leakage)
        df['clicked_time'] = df['DateTime'].where(df['is_click'] == 1)
        df['last_click_time'] = (df.sort_values(by="DateTime").groupby('user_id')['clicked_time'].transform(lambda s: s.ffill().shift()))
        df['hours_since_last_click'] = (df['DateTime'] - df['last_click_time']).dt.total_seconds() / 3600
        df.drop(columns=['clicked_time','last_click_time'], inplace=True)

        # user's number of unique values of 'webpage_id', 'product_category_1', 'campaign_id', 'product' (without leakage)
        def cumulative_unique_count(values):
            """
            For a 1D array/Series of values, return an array of
            'number of unique values so far (before this row)'.
            """
            seen = set()
            result = []
            for val in values:
                result.append(len(seen))
                seen.add(val)
            return result

        cols = ['webpage_id', 'product_category_1', 'campaign_id', 'product']
        for col in cols:
            df['user_distinct_' + col] = df.sort_values("DateTime").groupby('user_id', group_keys=False)[col].apply(cumulative_unique_count)

        # # Generate time-based features
        # df['Day'] = df['DateTime'].dt.day
        # df['Hour'] = df['DateTime'].dt.hour
        # df['Minute'] = df['DateTime'].dt.minute
        # df['weekday'] = df['DateTime'].dt.weekday

        # cols_to_fill = ["Day", "Hour", "Minute", "weekday"]   
        # self.logger.info(f"Filling missing values for columns: {cols_to_fill}")
        # df = self.fill_with_ffill_bfill_user(df, cols_to_fill)
        # #(df.groupby("user_id",observed = True)[cols_to_fill]
        # #    .transform(lambda x: x.ffill().bfill())
        # #    .infer_objects(copy=False)
        # #)
        # colls_to_fill_nas = df[cols_to_fill].isna().sum()
        # if colls_to_fill_nas.sum() > 0:
        #     self.logger.warning(f"Still missing values in columns: {colls_to_fill_nas.sum()}")
        #     df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].mode().iloc[0])
        # self.logger.info("Filled missing values with forward/backward fill.")
        # # Fill missing values
        # if self.use_missing_with_mode:
        #     df = self.fill_missing_values(df, use_mode=True)

        # # Generate campaign-based features
        # df['start_date'] = df.groupby('campaign_id', observed=True)['DateTime'].transform('min')
        # df['campaign_duration'] = df['DateTime'] - df['start_date']
        # df['campaign_duration_hours'] = df['campaign_duration'].dt.total_seconds() / (3600)
        # df['campaign_duration_hours'] = df['campaign_duration_hours'].fillna(
        #     df.groupby('campaign_id', observed=True)['campaign_duration_hours'].transform(lambda x: x.mode().iloc[0])
        #     )
        # df['campaign_duration_hours'] = pd.to_numeric(df['campaign_duration_hours'], errors='coerce')
        # self.logger.info(f'missing values in campaign_duration_hours: {df["campaign_duration_hours"].isna().sum()}')
        

        # # Drop unnecessary columns
        # df.drop(columns=['DateTime', 'start_date', 'campaign_duration', 'session_id', 'user_id', 'user_group_id'], inplace=True)
      
        # if self.catb:
        #     self.determine_categorical_features(df)
        # # One-hot encoding if `get_dumm` is True
        # df['campaign_duration_hours'] = df.groupby('webpage_id', observed=True)['campaign_duration_hours'].transform(
        #     lambda x: x.ffill().bfill() if not x.mode().empty else x.fillna(0))
        

        # self.logger.info(f'missing values in campaign_duration_hours after: {df["campaign_duration_hours"].isna().sum()}')

        # if self.use_dummies:
        #     columns_to_d = ["product", "campaign_id", "webpage_id", "product_category", "gender"]
        #     df = pd.get_dummies(df, columns=columns_to_d)

        return df
       
    """
    ╔═╗┬─┐┌─┐┌─┐┬─┐┌─┐┌─┐┌─┐┌─┐┌─┐
    ╠═╝├┬┘├┤ ├─┘├┬┘│ ││  ├┤ └─┐└─┐
    ╩  ┴└─└─┘┴  ┴└─└─┘└─┘└─┘└─┘└─┘

    """
    
    def preprocess(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:

        df_train = self.drop_completely_empty(df_train)

        df_train = self.drop_na_session_id_or_is_click(df_train)

        df_train = self.drop_dup_session_id(df_train)

        df_test = self.decrease_test_user_group_id(df_test) 

        df_test = self.replace_test_user_depth_to_training(df_train, df_test)

        df = self.concat_train_test(df_train, df_test)

        self.logger.info(f"Total number of missing values in the joint dataset: {df.isna().sum().sum()}")
        df = self.deterministic_fill(df)
        self.logger.info(f"Total number of missing values in the joint dataset after deterministic_fill: {df.isna().sum().sum()}")

        df_train, df_test = self.split_to_train_test(df)   
        # if "DateTime" in df_clean.columns:
        #     df_clean["DateTime"] = pd.to_datetime(df_clean["DateTime"], errors="coerce")

        # if self.remove_outliers:
        #     df_clean = self.remove_outliers(df_clean)

        # if self.fillna:
        #     df_clean = self.fill_missing_values(df_clean)

        # df_clean = self.feature_generation(df_clean)

        cols_to_target_encode = [c for c in df_train.columns if c not in ["session_id", "DateTime", "is_click"]]

        df_train, df_test = self.add_target_encoding(df_train, df_test, cols_to_target_encode)

        # X = df_clean.drop(columns=["is_click"])
        # y = df_clean["is_click"]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # # Create stratified folds for train set

        # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # self.feature_generation(df_train)

        # fold_datasets = []

        # for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        #     X_train_fold = X_train.iloc[train_idx]
        #     y_train_fold = y_train.iloc[train_idx]
        #     X_val_fold = X_train.iloc[val_idx]
        #     y_val_fold = y_train.iloc[val_idx]
        #     fold_datasets.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        #     self.callback({
        #     f"fold_{fold}_train_dataset": X_train_fold,
        #     f"fold_{fold}_val_dataset": X_val_fold,
        #     f"fold_{fold}_train_target": y_train_fold,
        #     f"fold_{fold}_val_target": y_val_fold
        #     })
        # self.callback({
        #     "X_train": X_train,
        #     "X_test": X_test,
        #     "y_train": y_train,
        #     "y_test": y_test
        # })
        # self.logger.info("Created stratified folds for training data.")

        # return df_clean, X_train, X_test, y_train, y_test, fold_datasets
    
    """
     ____                  
    / ___|  __ ___   _____ 
    \___ \ / _` \ \ / / _ \
     ___) | (_| |\ V /  __/
    |____/ \__,_| \_/ \___|
                        
    """

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
    parser.add_argument("--train_csv_path", type=str, default="data/raw/train_dataset_full.csv", help="Path to the input train CSV file")
    parser.add_argument("--test_csv_path", type=str, default="data/raw/X_test_1st.csv", help="Path to the test CSV file")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save the output data")
    parser.add_argument("--remove-outliers", action="store_true", help="Flag to remove outliers")
    parser.add_argument("--fillna", action="store_true", help="Flag to fill missing values")
    parser.add_argument("--use-dummies", action="store_true", help="Flag to apply pd.get_dummies")
    parser.add_argument("--use-missing-with-mode", action="store_true", help="Flag to fill missing values with mode")
    parser.add_argument("--save-as-pickle", action="store_true", default=True, help="Flag to save as Pickle instead of CSV")
    parser.add_argument("--fill-cat", action="store_true", help="Flag to fill categorical columns")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        output_path=Path(args.output_path),
        remove_outliers=args.remove_outliers,
        fillna=args.fillna,
        use_dummies=args.use_dummies,
        save_as_pickle=args.save_as_pickle
    )

    # df = preprocessor.load_data(Path(args.csv_path))
    df_train, df_test = preprocessor.load_all_data(Path(args.train_csv_path), Path(args.test_csv_path))
    df_clean, X_train, X_test, y_train, y_test, fold_datasets = preprocessor.preprocess(df_train, df_test)
    preprocessor.save_data(df_clean, X_train, X_test, y_train, y_test, fold_datasets)
