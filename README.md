# YDATA-kaggle-assignment
# Click-Through Rate (CTR) Prediction Project

## Project Overview
This project implements a machine learning pipeline for predicting click-through rates in online advertising using user interaction data. Our goal is to build and evaluate models that can accurately predict the likelihood of users clicking on advertisements, helping optimize ad placements and campaign effectiveness.

## Project Structure
```
project/
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned and preprocessed data
├── models/                 # Saved model files
├── notebooks/
│   └── ctr_analysis.ipynb  # Exploratory Data Analysis
├── app/                    # Application code
└── src/
    ├── preprocess.py      # Data preprocessing pipeline
    ├── features.py        # Feature engineering
    ├── train.py           # Model training
    ├── predict.py         # Model prediction
    └── results.py         # Results analysis
```

## Dataset Description
Our dataset includes comprehensive user interaction records with the following features:

### User Features
- User Demographics (age, gender)
- User Behavior Metrics
- City Development Index

### Session Features
- Session ID
- DateTime
- Webpage Information

### Product Features
- Product Categories
- Campaign Details

### Target Variable
- Click Status (is_click)

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/shaye3/YDATA-kaggle-assignment.git
cd YDATA-kaggle-assignment
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage Guide

### Data Preprocessing
Clean and prepare the raw data:
```bash
python preprocess.py --csv_path "data/raw/train_dataset_full.csv"
```

### Model Training
Train the Random Forest model:
```bash
python train.py -m rf
```

## Model Architecture

### Random Forest Classifier
- **Features**: User demographics, product categories, session data
- **Hyperparameters**:
  - max_depth: 30
  - min_samples_leaf: 10
  - class_weight: balanced
- **Performance Metrics**:
  - Precision
  - Recall
  - F1 Score

## Project Pipeline
1. Data Preprocessing
   - Handle missing values
   - Remove duplicates
   - Feature standardization
2. Feature Engineering
   - One-hot encoding for categorical variables
   - Feature scaling
3. Model Training
   - Cross-validation
   - Hyperparameter tuning
4. Evaluation
   - Performance metrics calculation
   - Model validation

## Contributing
1. Fork the repository
2. Create your feature branch:
```bash
git checkout -b feature/YourFeature
```

3. Commit your changes:
```bash
git commit -m 'Add some feature'
```

4. Push to the branch:
```bash
git push origin feature/YourFeature
```

5. Create a new Pull Request

## Team Members
- Shay
- Omer
- Maor
- Yonatan