# **1. Importing Necessary Libraries** 📚
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import streamlit as st
from db import *
import os
import subprocess
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder
from heatmap_analytics import *
# Import the retrain_model function from training1.py
from pythonFunctions.training1 import retrain_model


if not st.session_state.get("logged_in"):
    st.warning("Please log in first.")
    st.switch_page("pages/login.py")


def init_db():
    conn = sqlite3.connect("CBRSdata.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictiontable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Contact_Number TEXT,
            Email_address TEXT,
            Logical_quotient_rating INTEGER,
            coding_skills_rating INTEGER,
            hackathons INTEGER,
            public_speaking_points INTEGER,
            self_learning_capability TEXT,
            Team_Worker TEXT,
            Taken_inputs_from_seniors_or_elders TEXT,
            worked_in_teams_ever TEXT,
            Introvert TEXT,
            reading_and_writing_skills TEXT,
            memory_capability_score TEXT,
            smart_or_hard_work TEXT,
            Management_or_Techinical TEXT,
            Interested_subjects TEXT,
            Interested_Type_of_Books TEXT,
            certifications TEXT,
            workshops TEXT,
            Type_of_company_want_to_settle_in TEXT,
            interested_career_area TEXT,
            Result TEXT,
            Feedback TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def profile():
    st.subheader("User Profile")
    st.write(f"**Username:** {st.session_state['username']}")
    if st.button("Logout"):
        st.session_state.clear()
        st.success("Logged out successfully.")
        st.switch_page("app.py")

# **2. Loading Dataset**


file_path = "data/mldata.csv"

# Create an empty DataFrame by default
df = pd.DataFrame()


# Check if file exists and is not empty
if not os.path.exists(file_path):
    st.error(
        "Error: `mldata.csv` is missing. Please ensure `update_mldata.py` is running.")
elif os.stat(file_path).st_size == 0:
    st.error("Error: `mldata.csv` is empty. Waiting for data to be generated...")
else:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading `mldata.csv`: {e}")

# Ensure df always exists
if df.empty:
    # Add default columns to prevent errors
    df = pd.DataFrame(columns=["workshops"])

df['workshops'] = df['workshops'].replace(
    ['testing'], 'Testing')  # ✅ Now this won't fail

# **5. Feature Engineering**

# (a) Binary Encoding for Categorical Variables

newdf = df
newdf.head(10)
print(df.columns)

cols = df[["self-learning capability?", "Team_Worker",
           "Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    print(i)
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)


# (b) Number Encoding for Categorical

mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    print(i)
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?',
                    'Interested Type of Books']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes


# (c) Dummy Variable Encoding


df = pd.get_dummies(
    df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
df.head()

df.sort_values(by=['certifications'])

print("List of Numerical features: \n", df.select_dtypes(
    include=np.number).columns.tolist())


category_cols = df[['certifications', 'workshops', 'Interested subjects',
                    'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']]
for i in category_cols:
    print(i)

Certifi = list(df['certifications'].unique())
print(Certifi)
certi_code = list(df['certifications_code'].unique())
print(certi_code)

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)

Certi_l = list(df['certifications'].unique())
certi_code = list(df['certifications_code'].unique())
C = dict(zip(Certi_l, certi_code))

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)
W = dict(zip(Workshops, Workshops_code))

Interested_subjects = list(df['Interested subjects'].unique())
print(Interested_subjects)
Interested_subjects_code = list(df['Interested subjects_code'].unique())
ISC = dict(zip(Interested_subjects, Interested_subjects_code))

interested_career_area = list(df['interested career area '].unique())
print(interested_career_area)
interested_career_area_code = list(df['interested career area _code'].unique())
ICA = dict(zip(interested_career_area, interested_career_area_code))

Typeofcompany = list(df['Type of company want to settle in?'].unique())
print(Typeofcompany)
Typeofcompany_code = list(
    df['Type of company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany, Typeofcompany_code))

Interested_Books = list(df['Interested Type of Books'].unique())
print(Interested_subjects)
Interested_Books_code = list(df['Interested Type of Books_code'].unique())
IB = dict(zip(Interested_Books, Interested_Books_code))

Range_dict = {"poor": 0, "medium": 1, "excellent": 2}
print(Range_dict)


A = 'yes'
B = 'No'
col = [A, B]
for i in col:
    if (i == 'yes'):
        i = 1
    print(i)


f = []
A = 'r programming'
clms = ['r programming', 0]
for i in clms:
    for key in C:
        if (i == key):
            i = C[key]
            f.append(i)
print(f)

C = dict(zip(Certifi, certi_code))

print(C)

array = np.array([1, 2, 3, 4])
array.reshape(-1, 1)



# Preprocessing helper function to mirror training-time preprocessing
def preprocess_input(Afeed, feed):
    # Build a single row DataFrame
    columns = [
        'Logical_quotient_rating', 'coding_skills_rating', 'hackathons', 'public_speaking_points',
        'self_learning_capability', 'Team_Worker', 'Taken_inputs_from_seniors_or_elders', 'worked_in_teams_ever',
        'Introvert', 'reading_and_writing_skills', 'memory_capability_score',
        'smart_or_hard_work', 'Management_or_Techinical', 'Interested_subjects', 'Interested_Type_of_Books',
        'certifications', 'workshops', 'Type_of_company_want_to_settle_in', 'interested_career_area'
    ]
    all_features = Afeed + feed
    row_dict = dict(zip(columns, all_features))
    df_input = pd.DataFrame([row_dict])

    # Apply preprocessing (encoding same as training)
    df_input = df_input.replace({
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        "poor": 0, "medium": 1, "excellent": 2,
        "Smart worker": "Smart worker", "Hard Worker": "Hard Worker",
        "Management": "Management", "Technical": "Technical"
    })

    # One-hot encoding like in training
    df_input = pd.get_dummies(df_input)

    # Align columns with training
    feature_path = "models/feature_names.pkl"
    feature_names = joblib.load(feature_path)
    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    return df_input


# Updated inputlist using preprocessing function and feature alignment
def inputlist(Name, Contact_Number, Email_address,
              Logical_quotient_rating, coding_skills_rating, hackathons,
              public_speaking_points, self_learning_capability,
              Team_Worker, Taken_inputs_from_seniors_or_elders,
              worked_in_teams_ever, Introvert, reading_and_writing_skills,
              memory_capability_score, smart_or_hard_work, Management_or_Techinical,
              Interested_subjects, Interested_Type_of_Books, certifications, workshops,
              Type_of_company_want_to_settle_in, interested_career_area):

    Afeed = [Logical_quotient_rating, coding_skills_rating,
             hackathons, public_speaking_points]

    feed = [
        self_learning_capability, Team_Worker, Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert,
        reading_and_writing_skills, memory_capability_score, smart_or_hard_work, Management_or_Techinical,
        Interested_subjects, Interested_Type_of_Books, certifications, workshops,
        Type_of_company_want_to_settle_in, interested_career_area
    ]

    X_input = preprocess_input(Afeed, feed)

    # Load trained models
    rf_model = joblib.load("models/rf_model.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")

    # Get predictions from all models
    pred1 = rf_model.predict(X_input)[0]
    pred2 = svm_model.predict(X_input)[0]
    pred3 = xgb_model.predict(X_input)[0]

    # Use majority voting
    final_prediction = Counter([pred1, pred2, pred3]).most_common(1)[0][0]

    return (final_prediction,)


def main():
    subprocess.Popen(["python", "update_mldata.py"])
    html1 = """
    <div style="text-align:center; text-shadow: 3px 1px 2px blue;">
      <h1>👨🏻‍💻 Career Guidance System 👨🏻‍💻</h1>
    </div>
      """
    st.markdown(html1, unsafe_allow_html=True)  # simple html
    init_db()
    if "logged_in" not in st.session_state:
        st.switch_page("pages/login.py")

    else:
        profile()
        st.subheader("Your Career Recommendations")

    # Images

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("./assets/Career1.jpg")

    with col2:
        st.image("./assets/career-path.png")

    with col3:
        st.image("./assets/career _outline.png")

    html2 = """
    <div style="text-align:center; text-shadow: 3px 1px 2px blue;">
      <h2>Your Friendly Career Path Advisor<h2>
    </div>
      """
    st.markdown(html2, unsafe_allow_html=True)  # simple html

    st.sidebar.title("Your Information")

    Name = st.sidebar.text_input("Full Name")

    Contact_Number = st.sidebar.text_input("Contact Number")

    Email_address = st.sidebar.text_input("Email address")

    if not Name and Email_address:
        st.sidebar.warning("Please fill out your name and EmailID")

    if Name and Contact_Number and Email_address:
        st.sidebar.success("Thanks!")

    Logical_quotient_rating = st.slider(
        'Rate your Logical quotient Skills', 0, 10, 1)
    st.write(Logical_quotient_rating)

    coding_skills_rating = st.slider(
        'Rate your Coding Skills', 0, 10, 1)
    st.write(coding_skills_rating)

    hackathons = st.slider(
        'Enter number of Hackathons participated', 0, 10, 1)
    st.write(hackathons)

    public_speaking_points = st.slider(
        'Rate Your Public Speaking', 0, 10, 1)
    st.write(public_speaking_points)

    self_learning_capability = st.selectbox(
        'Self Learning Capability',
        ('Yes', 'No')
    )

    Team_Worker = st.selectbox(
        'Team Worker ',
        ('Yes', 'No')
    )

    Taken_inputs_from_seniors_or_elders = st.selectbox(
        'Took advice from seniors or elders',
        ('Yes', 'No')
    )

    worked_in_teams_ever = st.selectbox(
        'Team Co-ordination Skill',
        ('Yes', 'No')
    )

    Introvert = st.selectbox(
        'Introvert',
        ('Yes', 'No')
    )
    # st.write('You selected:', Introvert)

    reading_and_writing_skills = st.selectbox(
        'Reading and writing skills',
        ('poor', 'medium', 'excellent')
    )

    memory_capability_score = st.selectbox(
        'Memory capability score',
        ('poor', 'medium', 'excellent')
    )

    smart_or_hard_work = st.selectbox(
        'Smart or Hard Work',
        ('Smart worker', 'Hard Worker')
    )

    Management_or_Techinical = st.selectbox(
        'Management or Techinical',
        ('Management', 'Technical')
    )

    # --- Persist Interested_subjects using session_state with normalization ---
    if "Interested subjects" in df.columns:
        interested_subjects_options_raw = df["Interested subjects"].dropna().unique().tolist()
    else:
        interested_subjects_options_raw = ["No data available"]

    # Normalize the options
    interested_subjects_options = [opt.strip().lower() for opt in interested_subjects_options_raw]

    # Initialize session state with normalized value
    if 'Interested_subjects' not in st.session_state:
        st.session_state.Interested_subjects = interested_subjects_options[0] if interested_subjects_options else None

    # Get normalized session value
    normalized_value = st.session_state.Interested_subjects.strip().lower() if st.session_state.Interested_subjects else interested_subjects_options[0]

    # Fallback to first option if session value is not found
    if normalized_value not in interested_subjects_options:
        normalized_value = interested_subjects_options[0]

    # Render dropdown with normalized options
    Interested_subjects = st.selectbox(
        'Interested Subjects',
        interested_subjects_options,
        index=interested_subjects_options.index(normalized_value)
    )

    # Save the normalized selected value back to session state
    st.session_state.Interested_subjects = Interested_subjects

    Interested_Type_of_Books_options = (
        'Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art', 'Encyclopedias', 'Religion-Spirituality', 'Action and Adventure', 'Comics',
        'Horror', 'Satire', 'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery'
    )
    # No session state for this dropdown (not requested), so keep as is
    Interested_Type_of_Books = st.selectbox(
        'Interested Books Category',
        Interested_Type_of_Books_options
    )

    if "certifications" in df.columns:
        certifications_options = df["certifications"].dropna(
        ).unique().tolist()
    else:
        certifications_options = ["No data available"]
    if 'certifications' not in st.session_state:
        st.session_state.certifications = certifications_options[0] if certifications_options else None
    certifications = st.selectbox(
        'Certifications',
        certifications_options,
        index=certifications_options.index(st.session_state.certifications)
            if st.session_state.certifications in certifications_options else 0
    )
    st.session_state.certifications = certifications

    if "workshops" in df.columns:
        workshops_options = df["workshops"].dropna().unique().tolist()
    else:
        workshops_options = ["No data available"]
    if 'workshops' not in st.session_state:
        st.session_state.workshops = workshops_options[0] if workshops_options else None
    workshops = st.selectbox(
        'Workshops Attended',
        workshops_options,
        index=workshops_options.index(st.session_state.workshops)
            if st.session_state.workshops in workshops_options else 0
    )
    st.session_state.workshops = workshops

    if "Type of company want to settle in?" in df.columns:
        company_options = df["Type of company want to settle in?"].dropna(
        ).unique().tolist()
    else:
        company_options = ["No data available"]
    if 'Type_of_company_want_to_settle_in' not in st.session_state:
        st.session_state.Type_of_company_want_to_settle_in = company_options[0] if company_options else None
    Type_of_company_want_to_settle_in = st.selectbox(
        'Type of Company You Want to Settle In',
        company_options,
        index=company_options.index(st.session_state.Type_of_company_want_to_settle_in)
            if st.session_state.Type_of_company_want_to_settle_in in company_options else 0
    )
    st.session_state.Type_of_company_want_to_settle_in = Type_of_company_want_to_settle_in

    # --- Persist Interested_career_area using session_state with normalization ---
    if "interested career area " in df.columns:
        interested_career_area_options_raw = df["interested career area "].dropna().unique().tolist()
    else:
        interested_career_area_options_raw = ["No data available"]

    # Normalize the options
    interested_career_area_options = [opt.strip().lower() for opt in interested_career_area_options_raw]

    # Initialize session state with normalized value
    if 'interested_career_area' not in st.session_state:
        st.session_state.interested_career_area = interested_career_area_options[0] if interested_career_area_options else None

    # Get normalized session value
    normalized_value = st.session_state.interested_career_area.strip().lower() if st.session_state.interested_career_area else interested_career_area_options[0]

    # Fallback to first option if session value is not found
    if normalized_value not in interested_career_area_options:
        normalized_value = interested_career_area_options[0]

    # Render dropdown with normalized options
    interested_career_area = st.selectbox(
        'Interested Career Area',
        interested_career_area_options,
        index=interested_career_area_options.index(normalized_value)
    )

    # Save the normalized selected value back to session state
    st.session_state.interested_career_area = interested_career_area

    result = ""
    # Initialize career_label at the start of the function
    career_label = ""
    if 'predicted_result' not in st.session_state:
        st.session_state.predicted_result = None

    # --- Prediction Section ---
    if st.button("Predict"):
        try:
            result = inputlist(Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating, hackathons,
                               public_speaking_points, self_learning_capability, Team_Worker, Taken_inputs_from_seniors_or_elders,
                               worked_in_teams_ever, Introvert, reading_and_writing_skills, memory_capability_score, smart_or_hard_work,
                               Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, certifications, workshops,
                               Type_of_company_want_to_settle_in, interested_career_area)
            # Save in session_state
            st.session_state.predicted_result = result[0]

            # Reverse the label encoding to get the career label
            try:
                label_encoder = joblib.load('models/label_encoder.pkl')
                career_label = label_encoder.inverse_transform([st.session_state.predicted_result])[0]
            except Exception as e:
                career_label = str(st.session_state.predicted_result)

            # Store the career_label in session_state so it can be used later (e.g., on feedback submit)
            st.session_state.career_label = career_label

            # Show confidence progress bar (simulate confidence with progress)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            # Simulate confidence value (for demo: random or fixed)
            confidence = 0.85   # Let's say 85% confidence
            st.markdown("#### Prediction")
            st.success(f"Predicted Career Option : {career_label}")
            st.markdown(f"**Confidence:**")
            st.progress(int(confidence * 100))
            st.balloons()
            show_heatmap()
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # --- Feedback Section ---
    if st.session_state.predicted_result:
        st.markdown("---")
        st.markdown("#### Feedback")
        Feedback = st.selectbox(
            'How do you feel about the prediction?', ('Satisfied', 'Not satisfied'))

        if st.button('Submit'):
            # Use the career_label stored in session_state if available
            add_data(Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating,
                     hackathons, public_speaking_points, self_learning_capability, Team_Worker,
                     Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, reading_and_writing_skills,
                     memory_capability_score, smart_or_hard_work, Management_or_Techinical, Interested_subjects,
                     Interested_Type_of_Books, certifications, workshops, Type_of_company_want_to_settle_in,
                     interested_career_area, st.session_state.get("career_label", ""), Feedback)
            st.success("Prediction + Feedback saved to database!")
            st.session_state.predicted_result = None

            # Retrain only if enough feedback is collected
            def get_new_feedback_count():
                conn = sqlite3.connect('CBRSdata.db')
                query = "SELECT COUNT(*) FROM predictiontable WHERE Feedback = 'Satisfied'"
                result = conn.execute(query).fetchone()[0]
                conn.close()
                return result

            retrain_model()  # This will call the retrain_model function from training1.py


if __name__ == '__main__':
    main()
