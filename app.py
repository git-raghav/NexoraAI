import streamlit as st
from database import init_database

st.set_page_config(
    page_title="ML Project",
    page_icon="🤖",
    layout="wide"
)

# Initialize database
init_database()

st.title("ML Project Dashboard")
st.write("""
Welcome to the ML Project Dashboard! This application allows you to:
* Train ML models on default datasets
* Upload and analyze your own datasets
* Visualize model performance and data drift
* Make predictions using trained models
""")

st.sidebar.success("Select a page above.")
