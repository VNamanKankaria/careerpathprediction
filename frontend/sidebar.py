import streamlit as st

def sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Recommendations"])

    return page
