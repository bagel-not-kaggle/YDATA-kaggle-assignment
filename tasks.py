from preprocess_old import Preprocessor
from train import Trainer
from predict import Predictor
import os

# Define paths
data_path = "data/marketing_campaign.csv"
model_path = "model/trained_model.pkl"
result_path = "result/predictions.csv"

# Preprocessing
preprocessor = Preprocessor()
data = preprocessor.load_data(data_path)
X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data)

# Training
trainer = Trainer()
trainer.train_model(X_train, y_train)
trainer.save_model(model_path)

# Prediction
predictor = Predictor(model_path)
predictions = predictor.predict(X_test)
predictor.save_results(predictions, result_path)