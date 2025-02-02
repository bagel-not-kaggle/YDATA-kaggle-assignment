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

    ########################################################################################################################################################
    
    """       
    ╦ ╦┌─┐┬  ┌─┐┌─┐┬─┐  ╔═╗┬ ┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌┌─┐
    ╠═╣├┤ │  ├─┘├┤ ├┬┘  ╠╣ │ │││││   │ ││ ││││└─┐
    ╩ ╩└─┘┴─┘┴  └─┘┴└─  ╚  └─┘┘└┘└─┘ ┴ ┴└─┘┘└┘└─┘

    """

    def load_data(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df_test = pd.read_csv("data/raw/X_test_1st.csv")
        
        self.logger.info(f"Loading file from: {csv_path}")
        return df, df_test
    
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
    
    def drop_completely_empty(self, df: pd.DataFrame) -> pd.DataFrame:
        '''drop completely empty rows'''
        df.dropna(how='all', inplace=True)
        return df
    
    def drop_session_id_or_is_click(self, df: pd.DataFrame) -> pd.DataFrame:
        '''drop rows missing session_id or is_click'''
        df.dropna(subset=["session_id","is_click"], inplace=True)
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



    def mode_target(self, df, columns, group_by_col="user_id"):
        df_temp = df.copy()
        result = pd.DataFrame(index=df_temp.index)
        
        for column in columns:
            mode = df_temp.groupby(group_by_col, observed=True)[column].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            result[column] = mode
        
        result = result.sample(frac=1)  # Shuffle rows
        return result
    
    def fill_with_mode(self,df, columns): 
        for column in columns:
            if column in df.columns:
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value.iloc[0])
        return df

    def _fill_with_median(self,df, columns): # Maybe groupby user_id
        for column in columns:
            if column in df.columns:
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
        return df
    
    def determine_categorical_features(self, df: pd.DataFrame, cat_features: list = None): ## For catboost
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
    
    ########################################################################################################################################################

    """
    ╔═╗┬┬  ┬    ╔╦╗┬┌─┐┌─┐┬┌┐┌┌─┐  ╦  ╦┌─┐┬  ┬ ┬┌─┐┌─┐
    ╠╣ ││  │    ║║║│└─┐└─┐│││││ ┬  ╚╗╔╝├─┤│  │ │├┤ └─┐
    ╚  ┴┴─┘┴─┘  ╩ ╩┴└─┘└─┘┴┘└┘└─┘   ╚╝ ┴ ┴┴─┘└─┘└─┘└─┘

    """

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


        # Apply mode-based filling if enabled
        if self.fill_cat:
            cat_cols_to_fill = ["product", "campaign_id", "webpage_id", "gender", "product_category", "user_group_id"]
            df = self.mode_target(df, cat_cols_to_fill, "user_id")

        cols_for_ffill_bfill = ["age_level", "city_development_index","var_1", "user_depth"]
        self.logger.info(f"Filling missing values with user_id for columns: {cols_for_ffill_bfill}")
        self.logger.info(f'Number of missing values before: {df[cols_for_ffill_bfill].isna().sum()}')
        df[cols_for_ffill_bfill] = self.mode_target(df, cols_for_ffill_bfill,"user_id")
        self.logger.info(f'Number of missing values after: {df[cols_for_ffill_bfill].isna().sum()}')

        if df[cols_for_ffill_bfill].isna().sum().sum() > 0:
            self.logger.warning(f'Still missing values in columns: {cols_for_ffill_bfill}')
            self.logger.info(f"Filling missing values with user_group_id for columns: {cols_for_ffill_bfill}")
            for col in cols_for_ffill_bfill:
                mask = df[col].isna()  # Identify rows where the value is still missing
                if mask.sum() > 0:
                    df.loc[mask, col] = self.mode_target(df, [col], "user_group_id").loc[mask, col]
            self.logger.info(f'Number of missing values after: {df[cols_for_ffill_bfill].isna().sum()}')

            if df[cols_for_ffill_bfill].isna().sum().sum() > 0:
                self.logger.warning(f'Still missing values in columns: {cols_for_ffill_bfill}')
                self.logger.info(f"Filling missing values with mode for columns: {cols_for_ffill_bfill}")
                df = self.fill_with_mode(df, cols_for_ffill_bfill)
                self.logger.info(f'Number of missing values after: {df[cols_for_ffill_bfill].isna().sum()}')

        
        return df

    ########################################################################################################################################################
    
    """
    ╔═╗┌─┐┌─┐┌┬┐┬ ┬┬─┐┌─┐  ╔═╗┌─┐┌┐┌┌─┐┬─┐┌─┐┌┬┐┬┌─┐┌┐┌
    ╠╣ ├┤ ├─┤ │ │ │├┬┘├┤   ║ ╦├┤ │││├┤ ├┬┘├─┤ │ ││ ││││
    ╚  └─┘┴ ┴ ┴ └─┘┴└─└─┘  ╚═╝└─┘┘└┘└─┘┴└─┴ ┴ ┴ ┴└─┘┘└┘

    """

    def smooth_ctr(self,data, target_col, alpha=10):
        """Smooths the CTR by adding a prior."""
        # 1) Compute clicks and views
        clicks = data.groupby(target_col)['is_click'].sum().rename(f'{target_col}_clicks')
        views = data.groupby(target_col)['session_id'].count().rename(f'{target_col}_views')
        
        # 2) Global CTR
        global_ctr = data['is_click'].mean()
        
        # 3) Calculate smoothed CTR
        #    (clicks + alpha * global_ctr) / (views + alpha)
        ctr = ((clicks + alpha * global_ctr) / (views + alpha)).rename(f'{target_col}_ctr')
        
        # 4) Merge back into data
        data = data.merge(ctr, how='left', on=target_col)
        
        # 5) Fill missing CTR with global CTR
        data[f'{target_col}_ctr'].fillna(global_ctr, inplace=True)
        
        return data


    def feature_generation(self, df: pd.DataFrame):
        df = df.copy()
        df = self.smooth_ctr(df, "user_id")
        df = self.smooth_ctr(df, "product")
        df = self.smooth_ctr(df, "campaign_id")
        colls_to_check = ['user_id_ctr', 'product_ctr', 'campaign_id_ctr']
        self.logger.info(f'missing values in columns: {colls_to_check} before: {df[colls_to_check].isna().sum()}')
        df[colls_to_check] = df[colls_to_check].fillna(df[colls_to_check].mean())
        self.logger.info(f'missing values in columns: {colls_to_check} after: {df[colls_to_check].isna().sum()}')

        # Generate time-based features
        df['Day'] = df['DateTime'].dt.day
        df['Hour'] = df['DateTime'].dt.hour
        df['Minute'] = df['DateTime'].dt.minute
        df['weekday'] = df['DateTime'].dt.weekday

        cols_to_fill = ["Day", "Hour", "Minute", "weekday"]   
        self.logger.info(f"Filling missing values with user_id for columns: {cols_to_fill}")
        df[cols_to_fill] = self.mode_target(df, cols_to_fill,"user_id")
        #(df.groupby("user_id",observed = True)[cols_to_fill]
        #    .transform(lambda x: x.ffill().bfill())
        #    .infer_objects(copy=False)
        #)
        colls_to_fill_nas = df[cols_to_fill].isna().sum()
        if colls_to_fill_nas.sum() > 0:
            self.logger.warning(f"Still missing values in columns: {colls_to_fill_nas.sum()}")
            df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].mode().iloc[0])
        self.logger.info("Filled missing values with forward/backward fill.")
        # Fill missing values
        if self.use_missing_with_mode:
            df = self.fill_missing_values(df, use_mode=True)

        # Generate campaign-based features
        df['start_date'] = df.groupby('campaign_id', observed=True)['DateTime'].transform('min')
        df['campaign_duration'] = df['DateTime'] - df['start_date']
        df['campaign_duration_hours'] = df['campaign_duration'].dt.total_seconds() / (3600)
        df['campaign_duration_hours'] = df['campaign_duration_hours'].fillna(
            df.groupby('campaign_id', observed=True)['campaign_duration_hours'].transform(lambda x: x.mode().iloc[0])
            )
        df['campaign_duration_hours'] = pd.to_numeric(df['campaign_duration_hours'], errors='coerce')
        self.logger.info(f'missing values in campaign_duration_hours: {df["campaign_duration_hours"].isna().sum()}')
        

        # Drop unnecessary columns
        df.drop(columns=['DateTime', 'start_date', 'campaign_duration', 'session_id', 'user_id'], inplace=True)
      
        if self.catb:
            self.determine_categorical_features(df)
        # One-hot encoding if `get_dumm` is True
        df['campaign_duration_hours'] = df.groupby('webpage_id', observed=True)['campaign_duration_hours'].transform(
            lambda x: x.ffill().bfill() if not x.mode().empty else x.fillna(0))
        

        self.logger.info(f'missing values in campaign_duration_hours after: {df["campaign_duration_hours"].isna().sum()}')

        if self.use_dummies:
            columns_to_d = ["product", "campaign_id", "webpage_id", "product_category", "gender"]
            df = pd.get_dummies(df, columns=columns_to_d)

        return df
    
    ########################################################################################################################################################
    
    """
    ╔═╗┬─┐┌─┐┌─┐┬─┐┌─┐┌─┐┌─┐┌─┐┌─┐
    ╠═╝├┬┘├┤ ├─┘├┬┘│ ││  ├┤ └─┐└─┐
    ╩  ┴└─└─┘┴  ┴└─└─┘└─┘└─┘└─┘└─┘

    """
    
    def preprocess(self, df_train: pd.DataFrame, df_test) -> tuple:
        df_train = self.drop_completely_empty(df_train).copy()

        df_train = self.drop_session_id_or_is_click(df_train)

        df_test = self.decrease_test_user_group_id(df_test) 

        df_test = self.replace_test_user_depth_to_training(df_train, df_test)

        df = self.concat_train_test(df_train, df_test)

        self.logger.info(f"Total number of missing values in the joint dataset: {df.isna().sum()}")
        df = self.deterministic_fill(df)
        self.logger.info(f"Total number of missing values in the joint dataset after deterministic_fill: {df.isna().sum()}")

        df_train, df_test = self.split_to_train_test(df)
        df_train["DateTime"] = pd.to_datetime(df_train["DateTime"], errors="coerce")
        df_test["DateTime"] = pd.to_datetime(df_test["DateTime"], errors="coerce")

        if self.remove_outliers:
            df_train = self.remove_outliers(df_train)
            df_test = self.remove_outliers(df_test)

        if self.fillna:
            df_train = self.fill_missing_values(df_train)
            df_test = self.fill_missing_values(df_test)

        df_train = self.feature_generation(df_train)
        df_test = self.feature_generation(df_test)

        X = df_train.drop(columns=["is_click"])
        y = df_train["is_click"]

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
            f"fold_{fold}_train_dataset": X_train_fold,
            f"fold_{fold}_val_dataset": X_val_fold,
            f"fold_{fold}_train_target": y_train_fold,
            f"fold_{fold}_val_target": y_val_fold
            })
        self.callback({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        })
        self.logger.info("Created stratified folds for training data.")

        return df_train, X_train, X_test, y_train, y_test, fold_datasets, df_test
    
    r"""
     ____                  
    / ___|  __ ___   _____ 
    \___ \ / _` | \ / / _ |
     ___) | (_| |\ V /  __/
    |____/ \__,_| \_/ \___|
                        
    """

    def save_data(self, df_train, X_train, X_test, y_train, y_test, fold_datasets, df_test):
        if self.save_as_pickle:
            df_train.to_pickle(self.output_path / "cleaned_data_Maor.pkl")
            X_train.to_pickle(self.output_path / "X_train.pkl")
            X_test.to_pickle(self.output_path / "X_test.pkl")
            y_train.to_pickle(self.output_path / "y_train.pkl")
            y_test.to_pickle(self.output_path / "y_test.pkl")
            df_test.to_pickle(self.output_path / "df_TEST_DoNotTouch.pkl")
            
            # Save folds
            for i, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(fold_datasets):
                X_train_fold.to_pickle(self.output_path / f"X_train_fold_{i}.pkl")
                y_train_fold.to_pickle(self.output_path / f"y_train_fold_{i}.pkl")
                X_val_fold.to_pickle(self.output_path / f"X_val_fold_{i}.pkl")
                y_val_fold.to_pickle(self.output_path / f"y_val_fold_{i}.pkl")

            self.logger.info(f"Saved preprocessed data, train-test split, and folds as Pickle to {self.output_path}")
        else:
            df_train.to_csv(self.output_path / "cleaned_data_Maor.csv", index=False)
            X_train.to_csv(self.output_path / "X_train.csv", index=False)
            X_test.to_csv(self.output_path / "X_test.csv", index=False)
            y_train.to_csv(self.output_path / "y_train.csv", index=False, header=True)
            y_test.to_csv(self.output_path / "y_test.csv", index=False, header=True)
            df_test.to_csv(self.output_path / "df_TEST_DoNotTouch.csv", index=False)
            
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

    df_train,df_test = preprocessor.load_data(Path(args.csv_path))
    df_train, X_train, X_test, y_train, y_test, fold_datasets,df_test = preprocessor.preprocess(df_train,df_test)
    preprocessor.save_data(df_train, X_train, X_test, y_train, y_test, fold_datasets,df_test)
