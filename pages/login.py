import streamlit as st
import sqlite3

def init_db():
    conn = sqlite3.connect("CBRSdata.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()  # ensure table is created


def set_page_config():
    st.set_page_config(page_title="User Authentication", page_icon="🔑", layout="wide")
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 10px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 8px;
        }
        .stRadio>label {
            font-size: 18px;
            font-weight: bold;
        }
        .logout-container {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

def signup():
    st.subheader("🌟 Create a New Account")
    new_username = st.text_input("🆔 Username")
    new_password = st.text_input("🔒 Password", type="password")
    if st.button("Sign Up", help="Click to create a new account"):
        conn = sqlite3.connect("CBRSdata.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_username, new_password))
            conn.commit()
            st.session_state["logged_in"] = True
            st.session_state["username"] = new_username
            st.success(f"✅ Account created successfully! Redirecting...")
            st.switch_page("app.py")
        except sqlite3.IntegrityError:
            st.error("⚠️ Username already exists. Choose a different one.")
        conn.close()

def login():
    st.subheader("🔑 Login Page")
    username = st.text_input("🆔 Username")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Login", help="Click to log in to your account"):
        conn = sqlite3.connect("CBRSdata.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"✅ Welcome, {username}!")
            st.switch_page("app.py")
        else:
            st.error("❌ Invalid username or password")

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.switch_page("pages/login.py")

def main():
    set_page_config()
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            st.button("🚪 Logout", on_click=logout, key="logout_btn")
    else:
        st.title("🚀 User Authentication Portal")
        option = st.radio("Select an option", ["Login", "Sign Up"], horizontal=True)
        st.divider()
        if option == "Login":
            login()
        else:
            signup()

if __name__ == "__main__":
    main()
