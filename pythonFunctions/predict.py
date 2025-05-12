# predict.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load and clean dataset
df = pd.read_csv("data/mldata.csv")
df = df.dropna()

# Encode target
df['Result'] = df['Result'].astype('category')
y = df['Result'].cat.codes
X = df.drop(columns=['Result'])

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(class_weight='balanced')
svm_model = SVC(class_weight='balanced', probability=True)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Evaluate
print("RandomForest:", accuracy_score(y_test, rf_model.predict(X_test)))
print("SVM:", accuracy_score(y_test, svm_model.predict(X_test)))
print("XGBoost:", accuracy_score(y_test, xgb_model.predict(X_test)))

# Save models
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(xgb_model, 'models/xgb_model.pkl')