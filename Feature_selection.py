from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from sklearn.metrics import f1_score, precision_recall_curve, auc
import numpy as np

X_train = pd.read_pickle('data/processed/X_train.pkl')
y_train = pd.read_pickle('data/processed/y_train.pkl')

selected_features_list = []
pr_auc_list = []
f1_list = []
n_features_list = np.arange(10, 42, 1)
with open('data/Hyperparams/best_params115.json', 'r') as f:
    sample_params = json.load(f)

X_val = pd.concat([pd.read_pickle('data/processed/X_val_fold_0.pkl'),
                   pd.read_pickle('data/processed/X_val_fold_1.pkl'),
                   pd.read_pickle('data/processed/X_val_fold_2.pkl')
                   ,pd.read_pickle('data/processed/X_val_fold_3.pkl'),
                   pd.read_pickle('data/processed/X_val_fold_4.pkl')])

y_val = pd.concat([pd.read_pickle('data/processed/y_val_fold_0.pkl'),
                    pd.read_pickle('data/processed/y_val_fold_1.pkl'),
                    pd.read_pickle('data/processed/y_val_fold_2.pkl')
                    ,pd.read_pickle('data/processed/y_val_fold_3.pkl'),
                    pd.read_pickle('data/processed/y_val_fold_4.pkl')])


cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

model = CatBoostClassifier(
    **sample_params,
    cat_features=cat_features,  # Initial categorical features
)

for k in n_features_list:
    # Use column indices for feature selection
    selected_features = model.select_features(
        X=X_train,
        y=y_train,
        eval_set=(X_val, y_val),
        num_features_to_select=k,
        train_final_model=False,
        features_for_select=list(range(X_train.shape[1])),
        algorithm="RecursiveByPredictionValuesChange",
        logging_level="Verbose",
        plot=False
    )
    
    # Get selected feature names
    selected_feature_names = selected_features['selected_features_names']
    selected_features_list.append(selected_feature_names)
    X_train_new = X_train[selected_feature_names]
    X_val_new = X_val[selected_feature_names]
    
    # Update categorical features based on selected features
    new_cat_features = [col for col in cat_features if col in selected_feature_names]
    
    # Create new model with updated categorical features
    model_new = CatBoostClassifier(
        **sample_params,
        cat_features=new_cat_features,
    )
    
    # Fit and evaluate
    model_new.fit(X_train_new, y_train, eval_set=(X_val_new, y_val), use_best_model=True)
    y_pred = model_new.predict(X_val_new)
    f1 = f1_score(y_val, y_pred)
    pr, rc, _ = precision_recall_curve(y_val, model_new.predict_proba(X_val_new)[:, 1])
    pr_auc = auc(rc, pr)
    
    pr_auc_list.append(pr_auc)
    f1_list.append(f1)

import pickle
# save the features list to disk
with open(r'data\Predictions\selected_features_list.pkl', 'wb') as f:
    pickle.dump(selected_features_list, f)
with open(r'data\Predictions\pr_auc_list.pkl', 'wb') as f:
    pickle.dump(pr_auc_list, f)
with open(r'data\Predictions\f1_list.pkl', 'wb') as f:
    pickle.dump(f1_list, f)