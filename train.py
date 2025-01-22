import argparse
import pandas as pd
import logging
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
#import optuna

class ModelTrainer:
    def __init__(self, data_dir: str, model_name: str = "catboost", cat_features: list = None):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.cat_features = cat_features

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self):

        self.logger.info(f"Loading preprocessed data from {self.data_dir}...")
        X_train = pd.read_pickle(self.data_dir / "X_train.pkl")
        y_train = pd.read_pickle(self.data_dir / "y_train.pkl").squeeze()
        return X_train, y_train

    def determine_categorical_features(self, X_train: pd.DataFrame):

        if self.cat_features:
            cat_features = [col for col in self.cat_features if col in X_train.columns]
        else:
            cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in cat_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category").cat.add_categories("missing").fillna("missing")

        self.logger.info(f"Categorical features: {cat_features}")
        return cat_features

    def cross_validate_model(self, X_train: pd.DataFrame, y_train: pd.Series, cat_features: list, cv: int = 5):

        if self.model_name == 'catboost':
            model = CatBoostClassifier(
                random_seed=42, verbose=0, eval_metric='F1',
                cat_features=cat_features, class_weights=[1, 10]
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.logger.info(f"Performing {cv}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            self.logger.info(f"Processing fold {fold + 1}...")

            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            self.logger.info(f"Validation set shape for fold {fold + 1}: {X_fold_val.shape}")

            model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), use_best_model=True)

            fold_score = model.best_score_['validation']['F1']
            fold_scores.append(fold_score)

            self.logger.info(f"Fold {fold + 1} F1 score: {fold_score}")

        mean_cv_score = sum(fold_scores) / len(fold_scores)
        self.logger.info(f"Mean cross-validation F1 score: {mean_cv_score}")
        return mean_cv_score

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, cat_features: list, n_trials: int = 50):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'border_count': trial.suggest_int('border_count', 32, 255)
            }

            model = CatBoostClassifier(
                **params,
                random_seed=42, verbose=0, eval_metric='F1',
                cat_features=cat_features, class_weights=[1, 10]
            )

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), use_best_model=False)

                scores.append(model.best_score_['validation']['F1'])

            return sum(scores) / len(scores)

        self.logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, cat_features: list, val_size: float = 0.2):
        if self.model_name == 'catboost':
            model = CatBoostClassifier(
                random_seed=42, verbose=100, eval_metric='F1',
                cat_features=cat_features, class_weights=[1, 10]
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.logger.info(f"Training {self.model_name} model...")
        X_train.drop(columns=['session_id', 'DateTime', 'user_id'], inplace=True, errors='ignore')

        X_train_final, X_valid, y_train_final, y_valid = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )

        self.logger.info(f"Training {self.model_name} model with validation set...")
        model.fit(X_train_final, y_train_final, eval_set=(X_valid, y_valid), use_best_model=True)

        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f'{self.model_name}_model.cbm'
        model.save_model(str(model_path))
        self.logger.info(f"Model saved to {model_path}")
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to the directory with preprocessed data files")
    parser.add_argument("--model_name", type=str, default="catboost", help="Name of the model to train (default: catboost)")
    parser.add_argument("--cat_features", type=str, nargs='+', help="List of categorical feature names or indices")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--tune", action="store_true", help="Flag to perform hyperparameter tuning")
    parser.add_argument("--cv", type=str, default=False, help="Flag to perform cross-validation")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials for hyperparameter tuning (default: 50)")
    args = parser.parse_args()

    trainer = ModelTrainer(data_dir=args.data_dir, model_name=args.model_name, cat_features=args.cat_features)

    X_train, y_train = trainer.load_data()

    cat_features = trainer.determine_categorical_features(X_train)

    if args.tune:
        best_params = trainer.hyperparameter_tuning(X_train, y_train, cat_features, n_trials=args.n_trials)

    if args.cv:
        trainer.cross_validate_model(X_train, y_train, cat_features, cv=args.folds)

    trainer.train_model(X_train, y_train, cat_features)

    trainer.logger.info("Training complete.")
