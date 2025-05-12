import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.subheader("Here are some nerdy analytics 😁")
st.text("Correlation Between Numerical Features")
def show_heatmap():
    try:
        # Connect to DB and load data 
        conn = sqlite3.connect('CBRSdata.db')
        df = pd.read_sql("SELECT * FROM predictiontable", conn)
        conn.close()

        # Select only numerical columns for correlation
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        if not numeric_df.empty:
            corr = numeric_df.corr()

            # Set a custom style
            sns.set(style="whitegrid")

            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(
                corr,
                square=True,
                annot=True,
                linewidths=0.5,
                center=0,
                cmap="Blues",
                annot_kws={"size": 8},
                cbar_kws={"shrink": 0.7},
                fmt=".2f",
                ax=ax
            )
            ax.set_title("Correlation Heatmap of Numerical Features", fontsize=13, pad=10)
            st.pyplot(fig)
        else:
            st.warning("No numerical columns found to generate correlation heatmap.")

    except Exception as e:
        st.error(f"Error loading data or generating heatmap: {e}")
