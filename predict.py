import pickle
import numpy as np

class Predictor:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)

    def save_results(self, predictions, file_path):
        np.savetxt(file_path, predictions, delimiter=',', fmt='%d')
