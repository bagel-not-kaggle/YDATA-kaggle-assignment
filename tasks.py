#tasks

from preprocess import Preprocessor
# import wandb
# from train import Trainer
# from predict import Predictor
import os

# wandb_run = wandb.init(project="shay-kaggle-competition")
# Define paths
data_path = "data/raw/train_dataset_full.csv"
model_path = "model/trained_model.pkl"
result_path = "result/predictions.csv"

# Preprocessing
preprocessor = Preprocessor()
data = preprocessor.load_data(data_path)
processed_data = preprocessor.preprocess_data(data)

# # Training
# trainer = Trainer()
# trainer.train_model(X_train, y_train)
# trainer.save_model(model_path)
#
# # Prediction
# predictor = Predictor(model_path)
# predictions = predictor.predict(X_test)
# predictor.save_results(predictions, result_path)