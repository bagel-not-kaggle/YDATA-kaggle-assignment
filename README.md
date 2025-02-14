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
<<<<<<< HEAD
│   ├── ctr_analysis.ipynb  # Exploratory Data Analysis
    └── preprocessed.ipynb  # Exploring our processed files
=======
│   └── ctr_analysis.ipynb  # Exploratory Data Analysis
>>>>>>> d9649fbb6f5d050a9eb6d56ee63254a8667d7539
├── app/                    # Application code
└── src/
    ├── preprocess.py      # Data preprocessing pipeline
    ├── features.py        # Feature engineering
    ├── train.py           # Model training
    ├── predict.py         # Model prediction
<<<<<<< HEAD
    ├── results.py         # Results analysis
    ├── tasks.py           # Automated workflow
=======
    └── results.py         # Results analysis
>>>>>>> d9649fbb6f5d050a9eb6d56ee63254a8667d7539
```

## Dataset Description
Our dataset includes comprehensive user interaction records with the following features:

<<<<<<< HEAD
## Full Features list:
session_id, DateTime, user_id, webpage_id,

product, product_category, campaign_id,

user_group_id, gender, age_level, user_depth, city_development_index,

*var_1*

**is_click**

=======
>>>>>>> d9649fbb6f5d050a9eb6d56ee63254a8667d7539
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

<<<<<<< HEAD
### Wildcard
- *var_1*

### Target Variable
- **Click Status (is_click)**
=======
### Target Variable
- Click Status (is_click)
>>>>>>> d9649fbb6f5d050a9eb6d56ee63254a8667d7539

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
<<<<<<< HEAD
3. Add your changes:
```bash
git add files or .
```
4. Commit your changes:
=======

3. Commit your changes:
>>>>>>> d9649fbb6f5d050a9eb6d56ee63254a8667d7539
```bash
git commit -m 'Add some feature'
```

<<<<<<< HEAD
5. Push to the branch:
=======
4. Push to the branch:
>>>>>>> d9649fbb6f5d050a9eb6d56ee63254a8667d7539
```bash
git push origin feature/YourFeature
```

<<<<<<< HEAD
6. Create a new Pull Request
=======
5. Create a new Pull Request
>>>>>>> d9649fbb6f5d050a9eb6d56ee63254a8667d7539

## Team Members
- Shay
- Omer
- Maor
- Yonatan
- Nicole
