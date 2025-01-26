# flow_train.py
import wandb
from prefect import flow, task
from train import ModelTrainer

def wandb_callback(metrics: dict):
    """Logs the provided dictionary of metrics to WandB."""
    wandb.log(metrics)

@task(name="train_model_task")
def train_model_task(folds_dir: str, test_file: str):
    """Task to train and evaluate the model with WandB logging."""
    # Initialize WandB
    run = wandb.init(
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
    
    # Create ModelTrainer with a callback
    trainer = ModelTrainer(
        folds_dir=folds_dir,
        test_file=test_file,
        callback=wandb_callback   # Pass the callback
    )

    # Train and evaluate
    model = trainer.train_and_evaluate()

    # Finalize WandB run
    wandb.finish()
    
    return model

@flow(name="train_catboost_flow")
def train_flow(folds_dir: str, test_file: str):
    """Flow to orchestrate the training process."""
    # Call the task within the flow
    train_model_task(folds_dir, test_file)

if __name__ == "__main__":
    train_flow("data/processed", "data/processed")
