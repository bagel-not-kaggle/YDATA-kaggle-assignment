import argparse
import pandas as pd
import logging
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna

class ModelTrainer:
    def __init__(self, folds_dir: str, test_file: str, model_name: str = "catboost"):
        self.folds_dir = Path(folds_dir)
        self.test_file = Path(test_file)
        self.model_name = model_name

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_fold_data(self, fold_index):
        X_train = pd.read_pickle(self.folds_dir / f"X_train_fold_{fold_index}.pkl")
        y_train = pd.read_pickle(self.folds_dir / f"y_train_fold_{fold_index}.pkl").squeeze()
        X_val = pd.read_pickle(self.folds_dir / f"X_val_fold_{fold_index}.pkl")
        y_val = pd.read_pickle(self.folds_dir / f"y_val_fold_{fold_index}.pkl").squeeze()
        return X_train, y_train, X_val, y_val

    def determine_categorical_features(self, X_train: pd.DataFrame):
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        return cat_features

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, cat_features: list, n_trials: int = 50):
        def objective(trial):
            params = {
                "iterations": 1000,
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "No"]),
                "class_weights": [1, 1 / trial.suggest_float("class_weight_ratio", 1.0, 10.0)],
                "eval_metric": "F1",
                "early_stopping_rounds": 100,
                "random_seed": 42,
                "verbose": 0,
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

            model = CatBoostClassifier(**params)

            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train
            )

            model.fit(
                X_train_sub,
                y_train_sub,
                cat_features=cat_features,
                eval_set=(X_val_sub, y_val_sub),
                early_stopping_rounds=50,
                use_best_model=True
            )

            y_pred_val = model.predict(X_val_sub)
            f1 = f1_score(y_val_sub, y_pred_val)
            return f1

        self.logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params

    def train_and_evaluate(self):
        self.logger.info(f"Loading fold data from: {self.folds_dir}")
        n_folds = len(list(self.folds_dir.glob("X_train_fold_*.pkl")))
        self.logger.info(f"Detected {n_folds} folds.")
        a = 0.06767396213210575
        best_f1 = 0
        best_model = None
        fold_scores = []

        for fold_index in range(n_folds):
            self.logger.info(f"Processing fold {fold_index + 1}...")

            X_train, y_train, X_val, y_val = self.load_fold_data(fold_index)

            cat_features = self.determine_categorical_features(X_train)

            if self.model_name == "catboost":
                model = CatBoostClassifier(
                random_seed=42,
                verbose=100,
                eval_metric='F1',
                cat_features=cat_features,
            #auto_class_weights='Balanced',
                max_depth=5,
            #colsample_bylevel=0.7,
                class_weights=[1, 1/a],
                bagging_temperature=0.4,
                grow_policy='SymmetricTree',
            # one_hot_max_size = 40,
            # learning_rate=0.1,
            #  subsample=.67,   #lower subsample showed progress
            #  max_leaves= 64, #only with lossguide
                bootstrap_type = "Bayesian", #Bayesian uses the posterior probability of the object 
                                            #to sample the trees in the growing process. Good for regularization and overfitting control.
            # bootstrap_type='Bernoulli', #Bernoulli is Stochastic Gradient Boosting on random subsets of features, faster and less overfitting
                early_stopping_rounds=100,
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

            y_val_pred = model.predict(X_val)
            fold_f1 = f1_score(y_val, y_val_pred)
            fold_scores.append(fold_f1)

            self.logger.info(f"Fold {fold_index + 1} F1 score: {fold_f1}")

            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_model = model

        avg_f1 = sum(fold_scores) / len(fold_scores)
        self.logger.info(f"Average F1 score across folds: {avg_f1}")
        self.logger.info(f"Best F1 score across folds: {best_f1}")

        self.logger.info(f"Loading test data from: {self.test_file}")
        X_test = pd.read_pickle(self.folds_dir / "X_test.pkl")
        y_test = pd.read_pickle(self.folds_dir / "y_test.pkl").squeeze()

        self.logger.info("Predicting on test set using the best model...")
        y_test_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)
        self.logger.info(f"F1 score on test set: {test_f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CatBoost using pre-saved folds.")
    parser.add_argument("--folds_dir", type=str, default="data/processed", help="Directory containing fold data")
    parser.add_argument("--test_file", type=str, default="data/processed", help="Directory containing test set data")
    parser.add_argument("--model_name", type=str, default="catboost", help="Name of the model to train (default: catboost)")
    parser.add_argument("--tune", action="store_true", help="Flag to perform hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials for hyperparameter tuning (default: 50)")

    args = parser.parse_args()

    trainer = ModelTrainer(folds_dir=args.folds_dir, test_file=args.test_file, model_name=args.model_name)

    if args.tune:
        X_train, y_train = pd.read_pickle(trainer.folds_dir / "X_train.pkl"), pd.read_pickle(trainer.folds_dir / "y_train.pkl").squeeze()
        cat_features = trainer.determine_categorical_features(X_train)
        trainer.hyperparameter_tuning(X_train, y_train, cat_features, n_trials=args.n_trials)

    trainer.train_and_evaluate()
