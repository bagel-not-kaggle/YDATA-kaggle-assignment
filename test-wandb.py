import wandb
import random
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate some ground truth and predictions for demonstration
# Replace these with your actual y_true and y_pred
y_true = [random.randint(0, 9) for _ in range(100)]  # Example ground truth
y_pred = [random.randint(0, 9) for _ in range(100)]  # Example predictions

# Start a new wandb run to track this script
run = wandb.init(
    # Set the wandb project where this run will be logged
    project="my-awesome-project",

    # Config dictionary
    config={
        "hidden_layer_sizes": [32, 64],
        "kernel_sizes": [3],
        "activation": "ReLU",
        "pool_sizes": [2],
        "dropout": 0.5,
        "num_classes": 10,
    }
)

# Simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multi-class

    # Log metrics to wandb
    wandb.log({
        "acc": acc,
        "loss": loss,
        "f1_score": f1  # Log F1 score
    })

    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    wandb.log({"classification_report": report})

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Log confusion matrix as an image to wandb
    wandb.log({"Confusion Matrix": cm})
    plt.close()

# [Optional] Finish the wandb run, necessary in notebooks
wandb.finish()