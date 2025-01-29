import wandb
import random
import logging
import pandas as pd
from catboost import CatBoostClassifier
from prefect import task, flow
from sklearn.model_selection import train_test_split
from pathlib import Path


"""
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
"""




# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start a new WandB run
wandb.init(
    project="CTR-project",
    #entity="maorblumberg-tel-aviv-university-org",
    settings=wandb.Settings(start_method="thread"),
    config={
        "model_name": "catboost",
        "val_size": 0.2,
        "learning_rate": 0.03,
        "iterations": 1000,
        "class_weights": [1, 10]
    }
)

# Extract config from WandB

# Log the start of the task
logger.info("Starting the training task...")

# Load data
data_path = "data/processed"  # Replace with your data path
model_path = "models/catboost_model.cbm"  # Replace with your model path
data_dir = Path(data_path)
X_train = pd.read_pickle(data_dir / "X_train.pkl")
y_train = pd.read_pickle(data_dir / "y_train.pkl").squeeze()

# Identify categorical features
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Ensure categorical features are properly set
for col in cat_features:
    if col in X_train.columns:
        X_train[col] = (
            X_train[col]
            .astype("category")
            .fillna("missing")  # Just fill NA values with "missing"
        )
config = wandb.config

# Log the training setup
logger.info(f"Training CatBoost model with val_size={config.val_size} and iterations={config.iterations}...")

# Set up and train the model
model = CatBoostClassifier(
    random_seed=42,
    verbose=100,
    eval_metric="F1",
    cat_features=cat_features,
    class_weights=config.class_weights,
    learning_rate=config.learning_rate,
    iterations=config.iterations,
)

# Split data into training and validation sets
X_train_final, X_valid, y_train_final, y_valid = train_test_split(
    X_train, y_train, test_size=config.val_size, random_state=42
)

# Train the model
model.fit(X_train_final, y_train_final, eval_set=(X_valid, y_valid), use_best_model=True)

# Log training metrics to WandB
wandb.log({
    "best_iteration": model.get_best_iteration(),
    "train_f1": model.best_score_["learn"]["F1"],
    "valid_f1": model.best_score_["validation"]["F1"]
})

# Save the model
model_dir = Path(model_path).parent
model_dir.mkdir(parents=True, exist_ok=True)
model.save_model(model_path)
logger.info(f"Model saved to {model_path}")

# Log the trained model as a WandB artifact
#artifact = wandb.Artifact("trained_model", type="model")
#artifact.add_file(model_path)
#wandb.log_artifact(artifact)

# Finish the WandB run
wandb.finish()

logger.info("Training task completed successfully.")
