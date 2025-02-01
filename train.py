import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import pickle

class Trainer:
    def __init__(self, X_train, y_train, n_trials,model_path, param_distributions=None):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None
        self.n_trials = n_trials
        self.model_path = model_path
        self.param_distributions = param_distributions


    def objective(self, trial):
        # Define the hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 10, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

        # Create the model with the suggested hyperparameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=111
        )

        # Perform cross-validation
        score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=make_scorer(f1_score)).mean()
        print(f"Training Average Score: {score}")
        return score

    def tune_hyperparameters(self, n_trials=3):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print(f"Bets score found: {study.best_value}")
        self.model = RandomForestClassifier(**study.best_params, random_state=111)
        return study.best_params

    def train_model(self, n_trials=3):
        best_params = self.tune_hyperparameters(n_trials)
        print(f"Best parameters found: {best_params}")
        self.model.fit(self.X_train, self.y_train)

    # def tune_hyperparameters(self, n_iter=100):
    #     random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.param_distributions,
    #                                        n_iter=n_iter, cv=5, scoring='f1', n_jobs=-1, random_state=42)
    #     random_search.fit(self.X_train, self.y_train)
    #     self.model = random_search.best_estimator_
    #     return random_search.best_params_

    # def train_model(self, n_iter=100):
    #     best_params = self.tune_hyperparameters(self.param_distributions, n_iter)
    #     print(f"Best parameters found: {best_params}")
    #     self.model.fit(self.X_train, self.y_train)

    # def model_evaluate(self, metric='f1'):
    #     score = cross_val_score(self.model, self.X_test, self.y_test, cv=5, scoring=make_scorer(metric))
    #     return score.mean()

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def run_training(self):
        self.train_model()
        self.save_model(self.model_path)
        # return self.model_ejson()
