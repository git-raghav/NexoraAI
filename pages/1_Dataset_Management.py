import streamlit as st
import pandas as pd
from database import get_database_connection
import os
from ml_utils import load_default_dataset

st.title("Dataset Management")

# Create tabs for different dataset operations
tab1, tab2, tab3 = st.tabs(["Default Datasets", "Upload Dataset", "View Datasets"])

with tab1:
    st.header("Default Datasets")
    default_dataset = st.selectbox(
        "Select a default dataset",
        ["iris"]
    )

    if st.button("Load Default Dataset"):
        X, y = load_default_dataset(default_dataset)
        df = pd.concat([X, y.rename('target')], axis=1)

        # Save to database
        conn = get_database_connection()
        if conn:
            cur = conn.cursor()

            # Save dataset to file
            os.makedirs("datasets", exist_ok=True)
            file_path = f"datasets/{default_dataset}.csv"
            df.to_csv(file_path, index=False)

            # Save to database
            cur.execute(
                "INSERT INTO datasets (name, description, file_path) VALUES (%s, %s, %s)",
                (default_dataset, f"Default {default_dataset} dataset", file_path)
            )
            conn.commit()
            cur.close()
            conn.close()

            st.success(f"Successfully loaded {default_dataset} dataset!")
            st.dataframe(df)

with tab2:
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    dataset_name = st.text_input("Dataset Name")
    dataset_description = st.text_area("Dataset Description")

    if uploaded_file and dataset_name and st.button("Upload Dataset"):
        df = pd.read_csv(uploaded_file)

        # Save to database
        conn = get_database_connection()
        if conn:
            cur = conn.cursor()

            # Save dataset to file
            os.makedirs("datasets", exist_ok=True)
            file_path = f"datasets/{dataset_name}.csv"
            df.to_csv(file_path, index=False)

            # Save to database
            cur.execute(
                "INSERT INTO datasets (name, description, file_path) VALUES (%s, %s, %s)",
                (dataset_name, dataset_description, file_path)
            )
            conn.commit()
            cur.close()
            conn.close()

            st.success("Dataset uploaded successfully!")
            st.dataframe(df)

with tab3:
    st.header("View Datasets")
    conn = get_database_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, description, created_at FROM datasets")
        datasets = cur.fetchall()
        cur.close()
        conn.close()

        if datasets:
            for dataset in datasets:
                with st.expander(f"Dataset: {dataset[1]}"):
                    st.write(f"ID: {dataset[0]}")
                    st.write(f"Description: {dataset[2]}")
                    st.write(f"Created at: {dataset[3]}")

                    # Load and display dataset preview
                    df = pd.read_csv(f"datasets/{dataset[1]}.csv")
                    st.dataframe(df.head())
