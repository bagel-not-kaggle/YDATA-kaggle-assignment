import argparse
import pandas as pd
import logging
from pathlib import Path
#import onehot encoding
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import precision_recall_curve, auc

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
                 callback=None, params=None, select_features=False,
                 features_path=None):
        self.folds_dir = Path(folds_dir)
        self.test_file = Path(test_file)
        self.model_name = model_name
        self.callback = callback
        self.params = params
        self.select_features = select_features
        self.features_path = features_path

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

        

        # Load and merge all fold datasets
        X_train_all = []
        X_val_all = []
        y_train_all = []
        y_val_all = []
        
        n_folds = len(list(self.folds_dir.glob("X_train_fold_*.pkl")))
        for fold_index in range(n_folds):
            X_train_fold, y_train_fold, X_val_fold, y_val_fold = self.load_fold_data(fold_index)
            X_train_all.append(X_train_fold)
            X_val_all.append(X_val_fold)
            y_train_all.append(y_train_fold)
            y_val_all.append(y_val_fold)
        
        # Concatenate all folds
        X_train = pd.concat(X_train_all, axis=0)
        X_val = pd.concat(X_val_all, axis=0)
        y_train = pd.concat(y_train_all, axis=0)
        y_val = pd.concat(y_val_all, axis=0)
        cat_features = self.determine_categorical_features(X_train)
        # Convert all values in categorical columns that consist of 34546.0 for example to string
        for col in cat_features:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)

        if self.select_features:

            with open("data/Hyperparams/best_params115.json", 'r') as f:
                best_params = json.load(f)
            model = CatBoostClassifier(cat_features = cat_features, **best_params)
            self.logger.info("Starting feature selection")
            selected_features = model.select_features(
                X=X_train,
                y=y_train,
                eval_set=(X_val, y_val),
                num_features_to_select=24,
                train_final_model= False,
                features_for_select=list(range(X_train.shape[1])),
                algorithm="RecursiveByShapValues",
                logging_level="Verbose",
                plot=False
            )
            self.optimized_features = selected_features['selected_features_names']
            self.logger.warning(f"Selected features: {self.optimized_features}")
        
        if self.select_features:
            #save best features as pickle
            with open(f'data/Hyperparams/best_features{run_id}.pkl', 'wb') as f:
                pickle.dump(self.optimized_features, f)

            X_train_optimized = X_train[self.optimized_features]
            X_val_optimized = X_val[self.optimized_features]
        else:
            X_train_optimized = X_train
            X_val_optimized = X_val
        
        cat_features = self.determine_categorical_features(X_train_optimized)

        def objective(trial):
            # Define hyperparameters to optimize
            params = {   
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.065, 0.15),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 20, 30),
                "random_strength": trial.suggest_float("random_strength", 1.5, 4.8),
                "rsm": trial.suggest_float("rsm", 0.6, 1.0),
                "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 8, 24),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
                "iterations": 1000,
                #"eval_metric": trial.suggest_categorical("eval_metric", ["F1", "PRAUC:type=Classic"]),
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

            ## Initialize the model
            model = CatBoostClassifier(**params,cat_features = cat_features)
            model.fit(X_train_optimized, y_train, 
                      eval_set=(X_val_optimized, y_val),
                       early_stopping_rounds=50,
                      use_best_model=True)
            y_pred = model.predict_proba(X_val_optimized)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, y_pred)
            score = auc(recall, precision)
            

            # Return the average PRAUC score across folds
            trial_data = {
                "trial_number": len(trials_data) + 1,
                "PRAUC_score": score,
                 }
            trials_data.append(trial_data)

            if self.callback:
                self.callback({
                    "mean_PRAUC": score,
                    "trial_number": trial_data["trial_number"],
                    "trial_params": params  # Key change: separate hyperparameters
                })

            return score



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
                "eval_metric": "PRAUC:type=Classic",
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
        best_PRAUC = 0
        best_model = None
        fold_scores_val = []
        fold_scores_train = []
        self.chosen_features = None
        if self.params is not None:
            #read params
            with open(self.params, 'r') as f:
                params = json.load(f)
        else:
            params = {'depth': 3, 'learning_rate': 0.12117083431119458, 
                      'l2_leaf_reg': 27.49102055289926, 'random_strength': 1.2079636934696745,
                        'grow_policy': 'SymmetricTree', 'bootstrap_type': 'MVS',
                        'iterations': 1000, 'eval_metric': 'PRAUC:type=Classic', 'auto_class_weights': 'Balanced',
                        'early_stopping_rounds': 100, 'random_seed': 42, 'verbose': 0}

        for fold_index in range(n_folds):
            self.logger.info(f"Processing fold {fold_index + 1}...")

            X_train_cv, y_train_cv, X_val_cv, y_val_cv = self.load_fold_data(fold_index)

            if self.features_path is not None:

                with open(self.features_path, 'rb') as f:
                    self.optimized_features = pickle.load(f)
                X_train_cv = X_train_cv[self.optimized_features]
                X_val_cv = X_val_cv[self.optimized_features]

                cat_features = self.determine_categorical_features(X_train_cv)
                self.logger.warning(f"Selected features: {self.optimized_features}")

            self.logger.warning(f"X_train_cv shape: {X_train_cv.shape}")
            cat_features = self.determine_categorical_features(X_train_cv)
            
            if self.model_name == "catboost":
                model = CatBoostClassifier(
                cat_features=cat_features,
                **params)

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
                base_PRAUC_scores = {}

                for name, model in base_models.items():
                    model.fit(X_train_cv, y_train_cv)
                    y_pred_base = model.predict(X_val_cv)
                    y_pred_train = model.predict(X_train_cv)
                    precision, recall, _ = precision_recall_curve(y_val_cv, y_pred_base)
                    prauc = auc(recall, precision)
                    base_PRAUC_scores[name] = prauc
                    self.logger.info(f"{name} prauc score: {prauc}")

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

            

            y_val_pred = model.predict_proba(X_val_cv)[:, 1]
            y_train_pred = model.predict_proba(X_train_cv)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val_cv, y_val_pred)
            fold_prauc = auc(recall, precision)
            precision, recall, _ = precision_recall_curve(y_train_cv, y_train_pred)
            fold_prauc_train = auc(recall, precision)
            fold_scores_val.append(fold_prauc)
            fold_scores_train.append(fold_prauc_train)

            self.logger.info(f"Fold val {fold_index + 1} prauc score: {fold_prauc}")
            self.logger.info(f"Fold train {fold_index + 1} prauc score: {fold_prauc_train}")


            if self.callback:
                self.callback({f"fold_val_{fold_index + 1}_prauc": fold_prauc, 
                                f"fold_train_{fold_index + 1}_prauc": fold_prauc_train})

            if fold_prauc > best_PRAUC:
                best_PRAUC = fold_prauc
                best_model = model
                
        
        avg_prauc_train = sum(fold_scores_train) / len(fold_scores_train)
        self.logger.info(f"Average PRAUC train score across folds: {avg_prauc_train}")

        avg_prauc_val = sum(fold_scores_val) / len(fold_scores_val)
        self.logger.info(f"Average PRAUC val score across folds: {avg_prauc_val}")
        self.logger.info(f"Best PRAUC val score across folds: {best_PRAUC}")

        if self.callback:
            self.callback({"average_PRAUC_val": avg_prauc_val, "best_prauc_val": best_PRAUC})
          #                  "fold_scores_train": fold_scores_train})

        self.logger.info(f"Loading test data from: {self.test_file}")
        
        
        

        self.logger.info("Predicting on test set using the final model on the REAL TEST (warning: do not touch)")
        X_test = pd.read_pickle(self.folds_dir / "X_test.pkl")
        X_train = pd.read_pickle(self.folds_dir / "X_train.pkl")
        y_train = pd.read_pickle(self.folds_dir / "y_train.pkl").squeeze()
        y_test = pd.read_pickle(self.folds_dir / "y_test.pkl").squeeze()
        # Merge al validation folds for creating X_val
        X_val = pd.concat([pd.read_pickle(self.folds_dir / f"X_val_fold_{fold_index}.pkl") for fold_index in range(n_folds)], axis=0)
        y_val = pd.concat([pd.read_pickle(self.folds_dir / f"y_val_fold_{fold_index}.pkl") for fold_index in range(n_folds)], axis=0)
        if self.select_features:
            X_train = X_train[self.optimized_features]
            X_val = X_val[self.optimized_features]
            X_test = X_test[self.optimized_features]
        
        self.logger.warning(f"X_train shape: {X_train.shape}")

        cat_features = self.determine_categorical_features(X_train)

        model_fin = CatBoostClassifier(cat_features = cat_features, **params)
        model_fin.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)         
        y_class = model_fin.predict(X_test)
        y_test_pred = model_fin.predict_proba(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_test_pred[:, 1])
        test_prauc = auc(recall, precision)

        if best_model is None:
            best_model = model  # Fallback if no fold improved the PRAUC

        if self.model_name == "catboost":
            predictions_val = pd.DataFrame(y_class, columns=['is_click'])
            predict_proba_val = pd.DataFrame(y_test_pred, columns=['is_click_proba_0', 'is_click_proba_1'])
            predictions_val.to_csv(f'data/predictions/predictions_val{self.model_name}.csv', index=False)
            predict_proba_val.to_csv(f'data/predictions/predictions_proba_val{self.model_name}.csv', index=False)

            # Ensure models directory exists before saving
            Path("models").mkdir(parents=True, exist_ok=True)

            model_fin.save_model(f'models/best_model_{self.model_name}.cbm')
            self.logger.info(f"Model saved at models/best_model_{self.model_name}.cbm")


        self.logger.info(f"PRAUC score on test set: {test_prauc}")
        if self.callback:
            self.callback({"test_prauc": test_prauc})
        
        return  {
        "avg_prauc_train": avg_prauc_train,
        "avg_prauc_val": avg_prauc_val,
        "best_prauc_val": best_PRAUC,
        "test_prauc": test_prauc,
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
    parser.add_argument("--feature_importance", action="store_true", help="Flag to perform feature selection")
    parser.add_argument("--features_path", type=str, default=None, help="Path to the selected features")
    parser.add_argument("--select_features", action="store_true", help="Flag to perform feature selection")

    args = parser.parse_args()


    
    trainer = ModelTrainer(folds_dir=args.folds_dir, test_file=args.test_file, 
                           model_name=args.model_name, params=args.params, 
                            select_features=args.select_features, features_path=args.features_path)

    if args.tune:
        X_train, y_train = pd.read_pickle(trainer.folds_dir / "X_train.pkl"), pd.read_pickle(trainer.folds_dir / "y_train.pkl").squeeze()
        cat_features = trainer.determine_categorical_features(X_train)
        trainer.hyperparameter_tuning(X_train, y_train, cat_features, n_trials=args.n_trials, run_id=args.run_id)


    if args.feature_importance:
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
    
    
