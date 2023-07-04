import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from transformation import X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}
model = lgb.train(params, train_data, valid_sets=test_data, num_boost_round=1000, early_stopping_rounds=100)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)

# Use SMOTE to oversample the minority class in the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#####
# Defining the LightGBM model
lgb_model = lgb.LGBMClassifier(bagging_freq=5, bagging_fraction=0.8, boosting_type="rf", feature_fraction=0.6, learning_rate=0.05, num_iterations=100)
# Training the model on the resampled dataset
lgb_model.fit(X_train_resampled, y_train_resampled)
lgb_pred = lgb_model.predict(X_test)

# Confusion matrix
print("Confusion Matrix : \n",confusion_matrix(y_test,lgb_pred))
# Evaluate the performance of the model
print(classification_report(y_test,lgb_pred))