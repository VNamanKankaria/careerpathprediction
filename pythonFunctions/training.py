import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def retrain_model():
    # Connect to the database
    conn = sqlite3.connect('CBRSdata.db')
    query = "SELECT * FROM predictiontable WHERE Feedback = 'Satisfied'"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No rows with 'Satisfied' feedback found. Skipping model retraining.")
        return

    # Drop unnecessary columns
    df = df.drop(columns=['name', 'contact', 'email',
                 'Feedback'], errors='ignore')

    # Ensure 'Result' exists
    if 'Result' not in df.columns:
        raise ValueError("The 'Result' column is missing in the data.")

    # Encode target column
    df['Result'] = df['Result'].astype('category').cat.codes

    # Separate features and target
    X = df.drop(columns=['Result'])
    y = df['Result']

    # Replace empty strings with NaN
    X = X.replace('', np.nan)

    # Convert all columns to numeric where possible
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill missing values
    label_encoders = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Final check for missing values
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs still present in X.")
        print(X.isnull().sum())
        raise ValueError(
            "Missing values still exist in X after preprocessing.")

    # Align target with input
    y = y.loc[X.index]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define models
    rf_model = RandomForestClassifier(class_weight='balanced')
    svm_model = SVC(class_weight='balanced', probability=True)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train models
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Accuracy
    print("RandomForest Accuracy:", accuracy_score(
        y_test, rf_model.predict(X_test)))
    print("SVM Accuracy:", accuracy_score(y_test, svm_model.predict(X_test)))
    print("XGBoost Accuracy:", accuracy_score(
        y_test, xgb_model.predict(X_test)))

    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)

    # Save models
    joblib.dump(rf_model, 'model/rf_model.pkl')
    joblib.dump(svm_model, 'model/svm_model.pkl')
    joblib.dump(xgb_model, 'model/xgb_model.pkl')

    print("Models saved successfully.")
