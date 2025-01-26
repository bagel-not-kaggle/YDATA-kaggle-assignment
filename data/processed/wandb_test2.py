import wandb
import argparse
import pandas as pd
import logging
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from prefect import task, flow

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
        return X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    @task(name="train_and_evaluate")
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

            model = CatBoostClassifier(
                random_seed=42,
                verbose=100,
                eval_metric='F1',
                cat_features=cat_features,
                class_weights=[1, 1/a],
                bagging_temperature=0.4,
                grow_policy='SymmetricTree',
                bootstrap_type="Bayesian",
                early_stopping_rounds=100,
            )

            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
            y_val_pred = model.predict(X_val)
            fold_f1 = f1_score(y_val, y_val_pred)
            fold_scores.append(fold_f1)

            # Log fold metrics to wandb
            wandb.log({
                f"fold_{fold_index+1}_f1": fold_f1,
                "current_fold": fold_index + 1
            })

            self.logger.info(f"Fold {fold_index + 1} F1 score: {fold_f1}")

            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_model = model

        avg_f1 = sum(fold_scores) / len(fold_scores)
        
        # Log overall metrics to wandb
        wandb.log({
            "average_f1": avg_f1,
            "best_f1": best_f1
        })

        self.logger.info(f"Average F1 score across folds: {avg_f1}")
        self.logger.info(f"Best F1 score across folds: {best_f1}")

        X_test = pd.read_pickle(self.folds_dir / "X_test.pkl")
        y_test = pd.read_pickle(self.folds_dir / "y_test.pkl").squeeze()

        y_test_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # Log test metrics to wandb
        wandb.log({
            "test_f1": test_f1
        })
        
        self.logger.info(f"F1 score on test set: {test_f1}")
        return best_model

@flow(name="train_catboost_flow")
def train_flow(folds_dir: str, test_file: str):
    wandb.init(
        project="ctr-prediction",
        settings=wandb.Settings(start_method="thread"),
        config={
            "model_name": "catboost",
            "class_weight_ratio": 0.06767396213210575,
            "bagging_temperature": 0.4,
            "grow_policy": "SymmetricTree",
            "bootstrap_type": "Bayesian"
        }
    )
    
    trainer = ModelTrainer(folds_dir=folds_dir, test_file=test_file)
    model = trainer.train_and_evaluate()
    wandb.finish()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CatBoost using pre-saved folds.")
    parser.add_argument("--folds_dir", type=str, default="data/processed", help="Directory containing fold data")
    parser.add_argument("--test_file", type=str, default="data/processed", help="Directory containing test set data")
    
    args = parser.parse_args()
    train_flow(args.folds_dir, args.test_file)
