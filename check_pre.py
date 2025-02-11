import argparse
import pandas as pd
import logging
from pathlib import Path
#import onehot encoding
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import precision_recall_curve, auc

from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
import pickle
import numpy as np
import json
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.ensemble import StackingClassifier

## Import X_train and y_train
X_train = pd.read_pickle('data/processed/X_train.pkl')
y_train = pd.read_pickle('data/processed/y_train.pkl')

#import fold data
X_train_fold_0 = pd.read_pickle('data/processed/X_train_fold_0.pkl')
y_train_fold_0 = pd.read_pickle('data/processed/y_train_fold_0.pkl')
X_train_fold_1 = pd.read_pickle('data/processed/X_train_fold_1.pkl')
y_train_fold_1 = pd.read_pickle('data/processed/y_train_fold_1.pkl')
X_train_fold_2 = pd.read_pickle('data/processed/X_train_fold_2.pkl')
y_train_fold_2 = pd.read_pickle('data/processed/y_train_fold_2.pkl')
X_train_fold_3 = pd.read_pickle('data/processed/X_train_fold_3.pkl')
y_train_fold_3 = pd.read_pickle('data/processed/y_train_fold_3.pkl')
X_train_fold_4 = pd.read_pickle('data/processed/X_train_fold_4.pkl')
y_train_fold_4 = pd.read_pickle('data/processed/y_train_fold_4.pkl')
print(X_train.head())
print(X_train_fold_0.head())
print(X_train_fold_1.info())