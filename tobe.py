import numpy as np
import pandas as pd
from pathlib import Path

# Load data
predictions_proba = pd.read_csv(Path("data/Predictions/predictions_probability.csv"), header=None)
predictions_proba = pd.DataFrame(predictions_proba, columns=['0', '1'])
y_test = pd.read_pickle('data/processed/y_test.pkl')

# We need to use only the column '1' from predictions_proba for the calculations
predictions_proba_1 = predictions_proba['1']

# Entropy calculation 
entropy = -np.sum(predictions_proba * np.log(predictions_proba + 1e-10) +  
                 (1-predictions_proba) * np.log(1-predictions_proba + 1e-10), axis=1)

# Brier score calculation
brier = (predictions_proba_1 - y_test) ** 2

# Calibration error
calibration_error = np.abs(predictions_proba_1 - y_test)

# Get mean metrics
print("Brier Score:", np.mean(brier))
print("Entropy:", np.mean(entropy)) 
print("Calibration Error:", np.mean(calibration_error))
