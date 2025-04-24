import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Function to update the graph
def update_graph():
    file_path = "data/mldata.csv"
    
    try:
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist before computing correlation
        relevant_cols = ['Logical quotient rating', 'hackathons', 'coding skills rating', 'public speaking points']
        df = df[relevant_cols] if all(col in df.columns for col in relevant_cols) else pd.DataFrame()

        if not df.empty:
            corr = df.corr()

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough data to display correlation matrix.")

    except Exception as e:
        st.error(f"Error loading `mldata.csv`: {e}")

# Dynamic UI Update
st.subheader("📊 Dynamic Correlation Heatmap")
st.text("This updates automatically when new data arrives.")

# Refresh the graph every few seconds
while True:
    update_graph()
    time.sleep(60)  # Refresh every 60 seconds
