#train

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import pickle

class Trainer:
    def __init__(self, X_train, y_train, param_distributions=None):
        self.model = RandomForestClassifier(random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.param_distributions = param_distributions


    def tune_hyperparameters(self, n_iter=100):
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.param_distributions,
                                           n_iter=n_iter, cv=5, scoring='f1', n_jobs=-1, random_state=42)
        random_search.fit(self.X_train, self.y_train)
        self.model = random_search.best_estimator_
        return random_search.best_params_

    def train_model(self, n_iter=100):
        best_params = self.tune_hyperparameters(self.param_distributions, n_iter)
        print(f"Best parameters found: {best_params}")
        self.model.fit(self.X_train, self.y_train)

    def model_evaluate(self, metric='f1'):
        score = cross_val_score(self.model, self.X_test, self.y_test, cv=5, scoring=make_scorer(metric))
        return score.mean()

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def run_training(self):
        self.train_model()
        self.save_model("model.pkl")
        # return self.model_ejson()
