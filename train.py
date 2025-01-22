from sklearn.ensemble import RandomForestClassifier
import pickle

class Trainer:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)