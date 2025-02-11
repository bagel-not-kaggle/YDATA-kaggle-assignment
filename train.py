import argparse
import pandas as pd
import logging
from pathlib import Path
#import onehot encoding
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
import pickle
import numpy as np
import json
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.ensemble import StackingClassifier


class ModelTrainer:
    def __init__(self, folds_dir: str, test_file: str, model_name: str = "catboost",
                 callback=None, params=None, best_features=None, select_features=False):
        self.folds_dir = Path(folds_dir)
        self.test_file = Path(test_file)
        self.model_name = model_name
        self.callback = callback
        self.params = params
        self.best_features = best_features
        self.select_features = select_features

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    """       
    ╦ ╦┌─┐┬  ┌─┐┌─┐┬─┐  ╔═╗┬ ┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌┌─┐
    ╠═╣├┤ │  ├─┘├┤ ├┬┘  ╠╣ │ │││││   │ ││ ││││└─┐
    ╩ ╩└─┘┴─┘┴  └─┘┴└─  ╚  └─┘┘└┘└─┘ ┴ ┴└─┘┘└┘└─┘

    """

    def load_fold_data(self, fold_index):
        X_train = pd.read_pickle(self.folds_dir / f"X_train_fold_{fold_index}.pkl")
        y_train = pd.read_pickle(self.folds_dir / f"y_train_fold_{fold_index}.pkl").squeeze()
        X_val = pd.read_pickle(self.folds_dir / f"X_val_fold_{fold_index}.pkl")
        y_val = pd.read_pickle(self.folds_dir / f"y_val_fold_{fold_index}.pkl").squeeze()
        return X_train, y_train, X_val, y_val

    def determine_categorical_features(self, X_train: pd.DataFrame):
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        return cat_features
    
    def fill_missing_with_mode(self, X): # For naive stacking Model
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns

        df_temp = X.copy()

        for col in categorical_cols:
            mode_value = df_temp.loc[df_temp[col] != "missing", col].mode()[0]
            mask = df_temp[col] == "missing"
            df_temp.loc[mask, col] = mode_value
        return df_temp
    
    
    """
    ╦ ╦┬ ┬┌─┐┌─┐┬─┐┌─┐┌─┐┬─┐┌─┐┌┬┐┌─┐  ╔╦╗┬ ┬┌┐┌┌─┐
    ╠═╣└┬┘├─┘├┤ ├┬┘├─┘├─┤├┬┘├─┤│││└─┐   ║ │ ││││├┤ 
    ╩ ╩ ┴ ┴  └─┘┴└─┴  ┴ ┴┴└─┴ ┴┴ ┴└─┘   ╩ └─┘┘└┘└─┘

    """

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, cat_features: list, n_trials: int = 50, run_id: str = "1"):

        trials_data = []

        
        if self.select_features:
            self.logger.warning("Feature selection is enabled. Performing feature selection before hyperparameter tuning.")
            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train, y_train,
                                                                               test_size=0.2, random_state=42, 
                                                                               stratify=y_train)
            cat_features = self.determine_categorical_features(X_train_sub)
            with open("data/Hyperparams/best_params110.json", 'r') as f:
                best_params = json.load(f)
            model = CatBoostClassifier(cat_features = cat_features, **best_params)
            self.logger.info("Starting feature selection")
            selected_features = model.select_features(
                X=X_train_sub,
                y=y_train_sub,
                eval_set=(X_val_sub, y_val_sub),
                num_features_to_select=20,
                train_final_model= False,
                features_for_select=list(range(X_train_sub.shape[1])),
                algorithm="RecursiveByPredictionValuesChange",
                logging_level="Verbose",
                plot=False
            )
            self.optimized_features = selected_features['selected_features_names']
            self.logger.warning(f"Selected features: {self.optimized_features}")

        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                
                "depth": trial.suggest_int("depth", 4, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 15, 25),
                "random_strength": trial.suggest_float("random_strength", 1.5, 5),
                "rsm": trial.suggest_float("rsm", 0.6, 1.0),
                "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 8, 25),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
                "iterations": 1000,
                "eval_metric": trial.suggest_categorical("eval_metric", ["F1", "PRAUC:type=Classic"]),
                "auto_class_weights": "Balanced",
                "early_stopping_rounds": 100,
                
                "random_seed": 42,
                "verbose": 0,
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.6, 1.5)
                params["grow_policy"] = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.5, 0.9)
                params["grow_policy"] = "SymmetricTree"

            if params['grow_policy'] == 'Depthwise':    
                params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 3, 18)
            elif params['grow_policy'] == 'Lossguide':
                params['max_leaves'] = trial.suggest_int("max_leaves", 45, 64)
            
            if self.callback:
                self.callback({"trial_params": params})

            # Initialize the model
            model = CatBoostClassifier(**params)

            # Cross-validation
            #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
            scores = []

            #for fold_index, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                # Create train-validation splits
                #X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                #y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Check if both classes are present in the training and validation sets
            for fold_index in range(5):
                self.logger.info(f"Processing fold {fold_index + 1}...")

                X_train_cv, y_train_cv, X_val_cv, y_val_cv = self.load_fold_data(fold_index)
                #self.best_features = cat``
                if self.best_features is not None:
                    X_train_cv = X_train_cv[self.best_features]
                    X_val_cv = X_val_cv[self.best_features]

                if len(np.unique(y_train_cv)) < 2 or len(np.unique(y_val_cv)) < 2:
                    self.logger.warning(f"Fold {fold_index}: Skipping due to only one class in y_train_cv or y_val_cv")
                    continue
                cat_features = self.determine_categorical_features(X_train_cv)
                
                if self.select_features:
                    X_train_cv = X_train_cv[self.optimized_features]
                    X_val_cv = X_val_cv[self.optimized_features]
                    cat_features = self.determine_categorical_features(X_train_cv)
            
                model.fit(
                    X_train_cv,
                    y_train_cv,
                    cat_features=cat_features,
                    eval_set=(X_val_cv, y_val_cv),
                    early_stopping_rounds=50,
                    use_best_model=True
                )

                # Predict on the validation set
                y_pred_val = model.predict(X_val_cv)

                # Calculate F1 score
                try:
                    score = f1_score(y_val_cv, y_pred_val)
                    scores.append(score)
                    self.logger.info(f"Fold {fold_index}: F1 score = {score}")
                except Exception as e:
                    self.logger.error(f"Error calculating F1 score on fold {fold_index}: {e}")
                    continue

            # Return the average F1 score across folds
            mean_score = float(np.mean(scores) if scores else 0.0)
            trial_data = {
                "trial_number": len(trials_data) + 1,
                "f1_score": mean_score,
                 }
            trials_data.append(trial_data)

            if self.callback:
                self.callback({
                    "mean_f1": mean_score,
                    "trial_number": trial_data["trial_number"],
                    "trial_params": params  # Key change: separate hyperparameters
                })

            return mean_score



        self.logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        

        self.logger.info(f"Best hyperparameters: {study.best_params}")
        # add to study.best_params the constant parameters
        constant_params = {
             "iterations": 1000,
                "auto_class_weights": "Balanced",
                "early_stopping_rounds": 100,
                "random_seed": 42,
                "verbose": 0,
             }
        best_params = {**study.best_params, **constant_params}
        if self.callback:
            self.callback({"best_params": pd.DataFrame([best_params])})

        #save best hyperparameters as pickle
        with open(f'data/Hyperparams/best_params{run_id}.json', 'w') as f:
            json.dump(best_params, f, indent=4)  # `indent=4` makes the file human-readable


        return best_params
    
    
    """
    ╔═╗┌─┐┌─┐┌┬┐┬ ┬┬─┐┌─┐  ╔═╗┌─┐┬  ┌─┐┌─┐┌┬┐┬┌─┐┌┐┌
    ╠╣ ├┤ ├─┤ │ │ │├┬┘├┤   ╚═╗├┤ │  ├┤ │   │ ││ ││││
    ╚  └─┘┴ ┴ ┴ └─┘┴└─└─┘  ╚═╝└─┘┴─┘└─┘└─┘ ┴ ┴└─┘┘└┘

    """
    
    """
    def feature_selection_rfecv(self, X_train, y_train, n_trials: int = 50, run_id: str = "1"):
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        best_params = self.hyperparameter_tuning #With other model not cat boost
        columns_to_d = ["product", "campaign_id", "webpage_id", "product_category", "gender","user_group_id"]
        X_train = pd.get_dummies(X_train, columns=columns_to_d) #Rfecv does not work with categorical features
        mod_best = <entermodel>.fit(X_train, y_train)

        # Apply RFE with the best CatBoost model
        rfecv = RFECV(estimator=mod_best, step=1, #number of features to remove at each iteration 
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=44), scoring='f1',n_jobs=-1)
        rfecv.fit(X_train, y_train)

        # Selected features
        selected_features = X_train.columns[rfecv.support_]
        print("Selected Features:", selected_features)
        return selected_features
    """
    
    def feature_selection(self, X_train, y_train, n_trials: int = 10, run_id: str = "1", tune=False):
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if tune:
            best_params = self.hyperparameter_tuning(X_train, y_train, categorical_cols, n_trials=n_trials, run_id=run_id)
        else:
            with open(self.params, 'r') as f:
                best_params = json.load(f)
        self.logger.info(f'Starting feature selection with params: {best_params}')
        best_params['random_seed'] = 46
        model = CatBoostClassifier(**best_params,)
        self.logger.info(f"Training model with best hyperparameters: {self.params}")
        # Train model with categorical features properly handled
        model.fit(X_train, y_train, cat_features=categorical_cols)
        self.logger.info(f"Model training complete.")
        # Get feature importance using CatBoost's native method
        feature_importance = model.get_feature_importance()
        self.logger.info(f"Feature importance: {feature_importance}")
        feature_names = X_train.columns
        
        # Select features based on importance threshold
        important_features = feature_names[feature_importance > np.mean(feature_importance)]
        
        print("Selected Features:", important_features)
        return important_features, feature_importance, feature_names


    """
    ╔╦╗┬─┐┌─┐┬┌┐┌
     ║ ├┬┘├─┤││││
     ╩ ┴└─┴ ┴┴┘└┘

    """    

    def train_and_evaluate(self):
        self.logger.info(f"Loading fold data from: {self.folds_dir}")
        n_folds = len(list(self.folds_dir.glob("X_train_fold_*.pkl")))
        self.logger.info(f"Detected {n_folds} folds.")
        a = 0.06767396213210575
        best_f1 = 0
        best_model = None
        fold_scores_val = []
        fold_scores_train = []
        if self.params is not None:
            #read params
            with open(self.params, 'r') as f:
                params = json.load(f)
        else:
            params = {'depth': 3, 'learning_rate': 0.12117083431119458, 
                      'l2_leaf_reg': 27.49102055289926, 'random_strength': 1.2079636934696745,
                        'grow_policy': 'SymmetricTree', 'bootstrap_type': 'MVS',
                        'iterations': 1000, 'eval_metric': 'F1', 'auto_class_weights': 'Balanced',
                        'early_stopping_rounds': 100, 'random_seed': 42, 'verbose': 0}

        for fold_index in range(n_folds):
            self.logger.info(f"Processing fold {fold_index + 1}...")

            X_train_cv, y_train_cv, X_val_cv, y_val_cv = self.load_fold_data(fold_index)
            #self.best_features = cat``
            if self.best_features is not None:
                X_train_cv = X_train_cv[self.best_features]
                X_val_cv = X_val_cv[self.best_features]

            cat_features = self.determine_categorical_features(X_train_cv)
            if self.select_features:
                self.logger.warning(f"Starting feature selection")
                
                model = CatBoostClassifier(cat_features = cat_features, **params)
                selected_features = model.select_features(
                    X=X_train_cv,
                    y=y_train_cv,
                    eval_set=(X_val_cv, y_val_cv),
                    num_features_to_select=20,
                    train_final_model= False,
                    features_for_select=list(range(X_train_cv.shape[1])),
                    algorithm="RecursiveByPredictionValuesChange",
                    logging_level="Verbose",
                    plot=False
                )
                X_train_cv = X_train_cv[selected_features['selected_features_names']]
                X_val_cv = X_val_cv[selected_features['selected_features_names']]
                cat_features = self.determine_categorical_features(X_train_cv)
                self.logger.warning(f"Selected features: {selected_features['selected_features_names']}")

            if self.model_name == "catboost":
                model = CatBoostClassifier(
                cat_features=cat_features,
                **params)

            ## Stacking model

            elif self.model_name == "stacking":
                X_train_cv = self.fill_missing_with_mode(X_train_cv)
                X_val_cv = self.fill_missing_with_mode(X_val_cv)
                X_test = pd.read_pickle(self.folds_dir / "X_test.pkl")
                y_test = pd.read_pickle(self.folds_dir / "y_test.pkl").squeeze()
                X_test = self.fill_missing_with_mode(X_test)
                print("X_train", X_train_cv.isnull().sum().sum())
                print("X_val",X_val_cv.columns.isnull().sum().sum())   
                #use get_dummies to convert categorical columns to numerical
                columns_to_onehot = ["product", "campaign_id", "webpage_id", "product_category", "gender","user_group_id"]
                onehot = OneHotEncoder()
                X_train_cv = onehot.fit_transform(X_train_cv[columns_to_onehot]).toarray()  # Convert to dense
                X_val_cv = onehot.transform(X_val_cv[columns_to_onehot]).toarray()          # Convert to dense
                X_test = onehot.transform(X_test[columns_to_onehot]).toarray()        # Convert to dense
                sgd = SGDClassifier(random_state=42, loss='log_loss', class_weight='balanced')
                lr = LogisticRegression(random_state=42, C = 0.1, class_weight = 'balanced', solver = 'liblinear', max_iter = 1000)
                cb = ComplementNB()
                gb = GaussianNB()
                
                base_models = {'SGDClassifier': sgd, 'LogisticRegression': lr, 'ComplementNB': cb, 'GaussianNB': gb}
                base_f1_scores = {}

                for name, model in base_models.items():
                    model.fit(X_train_cv, y_train_cv)
                    y_pred_base = model.predict(X_val_cv)
                    y_pred_train = model.predict(X_train_cv)
                    f1 = f1_score(y_val_cv, y_pred_base)
                    base_f1_scores[name] = f1
                    self.logger.info(f"{name} F1 score: {f1}")

                model = StackingClassifier(estimators=[('sgd', sgd), 
                                                       ('lr', lr),
                                                       ('cb', cb), 
                                                       ('gb', gb)], final_estimator=LogisticRegression(random_state=42, C = 0.1, class_weight = 'balanced', solver = 'liblinear', max_iter = 1000),
                                                       cv=5)

            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            if self.model_name == "catboost":
                model.fit(X_train_cv, y_train_cv, eval_set=(X_val_cv, y_val_cv), use_best_model=True)

            elif self.model_name == "stacking":
                model.fit(X_train_cv, y_train_cv)

            

            y_val_pred = model.predict(X_val_cv)
            y_train_pred = model.predict(X_train_cv)
            fold_f1 = f1_score(y_val_cv, y_val_pred)
            fold_f1_train = f1_score(y_train_cv, y_train_pred)
            fold_scores_val.append(fold_f1)
            fold_scores_train.append(fold_f1_train)

            self.logger.info(f"Fold val {fold_index + 1} F1 score: {fold_f1}")
            self.logger.info(f"Fold train {fold_index + 1} F1 score: {fold_f1_train}")


            if self.callback:
                self.callback({f"fold_val_{fold_index + 1}_f1": fold_f1, 
                                f"fold_train_{fold_index + 1}_f1": fold_f1_train})

            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_model = model
        
        avg_f1_train = sum(fold_scores_train) / len(fold_scores_train)
        self.logger.info(f"Average F1 train score across folds: {avg_f1_train}")

        avg_f1_val = sum(fold_scores_val) / len(fold_scores_val)
        self.logger.info(f"Average F1 val score across folds: {avg_f1_val}")
        self.logger.info(f"Best F1 val score across folds: {best_f1}")

        if self.callback:
            self.callback({"average_f1": avg_f1_val, "best_f1": best_f1})
          #                  "fold_scores_train": fold_scores_train})

        self.logger.info(f"Loading test data from: {self.test_file}")
        
        
        

        self.logger.info("Predicting on test set using the best modeland on the REAL TEST (warning: do not touch)")
        X_test = pd.read_pickle(self.folds_dir / "X_test.pkl")
        if self.best_features is not None:
            X_test = X_test[self.best_features]
        y_test = pd.read_pickle(self.folds_dir / "y_test.pkl").squeeze()
        y_test_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)

        if self.model_name == "catboost":
            predictions_val = pd.DataFrame(y_test_pred, columns=['is_click'])
            predictions_val.to_csv(f'data/predictions/predictions_val{self.model_name}.csv', index=False)

        self.logger.info(f"F1 score on test set: {test_f1}")
        if self.callback:
            self.callback({"test_f1": test_f1})
        
        return  {
        "avg_f1_train": avg_f1_train,
        "avg_f1_val": avg_f1_val,
        "best_f1": best_f1,
        "test_f1": test_f1,
        "fold_scores_train": fold_scores_train,
        "fold_scores_val": fold_scores_val
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CatBoost using pre-saved folds.")
    parser.add_argument("--folds_dir", type=str, default="data/processed", help="Directory containing fold data")
    parser.add_argument("--test_file", type=str, default="data/processed", help="Directory containing test set data")
    parser.add_argument("--model_name", type=str, default="catboost", help="Name of the model to train (default: catboost)")
    parser.add_argument("--tune", action="store_true", help="Flag to perform hyperparameter tuning")
    parser.add_argument("--run_id", type=str, default="1", help="Run ID for hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials for hyperparameter tuning (default: 50)")
    parser.add_argument("--train", action="store_true", help="Flag to train the model")
    parser.add_argument("--params", type=str, default=None, help="Hyperparameters for the model")
    parser.add_argument("--best_features", type=str, default=None, help="Best features for the model")
    parser.add_argument("--feature_selection", action="store_true", help="Flag to perform feature selection")
    parser.add_argument("--select_features", action="store_true", help="Flag to perform feature selection")

    args = parser.parse_args()

    

    if args.best_features:
        args.best_features = args.best_features.split(',')
    else:
        args.best_features = None
    
    trainer = ModelTrainer(folds_dir=args.folds_dir, test_file=args.test_file, 
                           model_name=args.model_name, params=args.params, best_features=args.best_features,
                            select_features=args.select_features)

    if args.tune:
        X_train, y_train = pd.read_pickle(trainer.folds_dir / "X_train.pkl"), pd.read_pickle(trainer.folds_dir / "y_train.pkl").squeeze()
        cat_features = trainer.determine_categorical_features(X_train)
        trainer.hyperparameter_tuning(X_train, y_train, cat_features, n_trials=args.n_trials, run_id=args.run_id)


    if args.feature_selection:
        trainer = ModelTrainer(
            folds_dir=args.folds_dir,
            test_file=args.test_file,
            model_name=args.model_name,
            params=args.params
        )
        X_train = pd.read_pickle(Path(args.folds_dir) / "X_train.pkl")
        y_train = pd.read_pickle(Path(args.folds_dir) / "y_train.pkl").squeeze()
        trainer.feature_selection(X_train, y_train, n_trials=args.n_trials, run_id=args.run_id)

    if args.train:
        trainer.train_and_evaluate()
    
    
