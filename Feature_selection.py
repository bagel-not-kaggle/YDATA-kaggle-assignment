from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

X_train = pd.read_pickle('data/processed/X_train.pkl')
y_train = pd.read_pickle('data/processed/y_train.pkl')


sample_params = {'depth': 4,
  'learning_rate': 0.12751986192358583,
  'l2_leaf_reg': 28.56605893525792,
  'random_strength': 1.4329403288787461,
  'grow_policy': 'SymmetricTree',
  'bootstrap_type': 'Bayesian',
  'bagging_temperature': 0.31033906089109137}


X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
#select cat_features as object columns and categorical columns
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

model = CatBoostClassifier(
    **sample_params,
    random_seed=42,
    verbose=0,
    eval_metric='F1',
    cat_features=cat_features,
    auto_class_weights='Balanced'
)

# Perform feature selection
selected_features = model.select_features(
    X=X_train_sub,
    y=y_train_sub,
    eval_set=(X_val_sub, y_val_sub),
    num_features_to_select=10,
    features_for_select=list(range(X_train_sub.shape[1])),
    algorithm="RecursiveByPredictionValuesChange",
    logging_level="Verbose",
    plot=True)