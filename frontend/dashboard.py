import streamlit as st
from frontend.sidebar import sidebar

def dashboard():
    st.title("Career Trend Dashboard")
    st.write("Analyze career trends from real-time data.")

if __name__ == "__main__":
    page = sidebar()
    
    if page == "Home":
        from frontend.home import home
        home()
    elif page == "Dashboard":
        dashboard()
    elif page == "Recommendations":
        from frontend.recommendations import recommendations
        recommendations()
