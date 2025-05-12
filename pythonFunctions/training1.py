import sqlite3
import pandas as pd
import random
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # Import SMOTE for class balancing

def retrain_model():
    try:
        # Connect to the SQLite database
        print("Connecting to the database...")
        conn = sqlite3.connect('CBRSdata.db')
        cursor = conn.cursor()

        # Fetch data from the predictiontable
        print("Fetching data from predictiontable...")
        cursor.execute("SELECT * FROM predictiontable")
        data = cursor.fetchall()

        # Check if data is fetched
        if not data:
            print("❌ No data found in predictiontable!")
            return
        else:
            print(f"Data fetched: {len(data)} rows")

        # Create a DataFrame
        columns = [
            'name', 'contact', 'email', 'Logical_quotient_rating', 'coding_skills_rating',
            'hackathons', 'public_speaking_points', 'self_learning_capability', 'Team_Worker',
            'Taken_inputs_from_seniors_or_elders', 'worked_in_teams_ever', 'Introvert',
            'reading_and_writing_skills', 'memory_capability_score', 'smart_or_hard_work',
            'Management_or_Techinical', 'Interested_subjects', 'Interested_Type_of_Books',
            'certifications', 'workshops', 'Type_of_company_want_to_settle_in',
            'interested_career_area', 'Result', 'Feedback'
        ]
        df = pd.DataFrame(data, columns=columns)
        print("Data fetched and DataFrame created.")

        # Separate features and target variable
        X = df.drop(columns=['Result', 'name', 'contact', 'email', 'Feedback'])
        y = df['Result']

        # Handle missing data and categorical encoding
        print("Handling missing data and encoding categorical variables...")
        X = X.replace('', pd.NA)
        X = pd.get_dummies(X)
        X.fillna(X.mean(), inplace=True)

        # Label encode the target variable
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # Train models with balanced class weights
        print("Training models...")
        rf_model = RandomForestClassifier(class_weight='balanced')
        svm_model = SVC(probability=True, class_weight='balanced')
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        rf_model.fit(X_train, y_train)
        svm_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        # Evaluate on test set
        from sklearn.metrics import classification_report
        print("Random Forest Report:")
        print(classification_report(y_test, rf_model.predict(X_test)))
        print("SVM Report:")
        print(classification_report(y_test, svm_model.predict(X_test)))
        print("XGBoost Report:")
        print(classification_report(y_test, xgb_model.predict(X_test)))

        # Retrain models on full data for final saving
        rf_model.fit(X_res, y_res)
        svm_model.fit(X_res, y_res)
        xgb_model.fit(X_res, y_res)

        # Save the feature names after training
        print("Saving feature names...")
        joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')  # Save feature names
        print("Feature names saved.")

        joblib.dump(label_encoder, 'models/label_encoder.pkl')  # Save the label encoder

        # Save models
        os.makedirs('models', exist_ok=True)
        try:
            joblib.dump(rf_model, 'models/rf_model.pkl')
            joblib.dump(svm_model, 'models/svm_model.pkl')
            joblib.dump(xgb_model, 'models/xgb_model.pkl')
            print("✅ Models trained and saved successfully to the 'models/' folder.")
        except Exception as e:
            print("❌ Error saving models:", e)

        # Output summary
        print(f"🧠 Training completed on {len(X_res)} rows.")
        print(f"📊 Target classes: {sorted(set(y_res))}")
    
    except Exception as e:
        print(f"❌ An error occurred: {e}")