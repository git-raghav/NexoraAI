import streamlit as st
import pandas as pd
from database import get_database_connection
import os
from ml_utils import load_default_dataset
import io

# Initialize session state for datasets if it doesn't exist
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

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

        # Save to session state
        st.session_state.datasets[default_dataset] = {
            'data': df,
            'description': f"Default {default_dataset} dataset",
            'created_at': pd.Timestamp.now()
        }

        st.success(f"Successfully loaded {default_dataset} dataset!")
        st.dataframe(df)

with tab2:
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    dataset_name = st.text_input("Dataset Name")
    dataset_description = st.text_area("Dataset Description")

    if uploaded_file and dataset_name and st.button("Upload Dataset"):
        df = pd.read_csv(uploaded_file)

        # Save to session state
        st.session_state.datasets[dataset_name] = {
            'data': df,
            'description': dataset_description,
            'created_at': pd.Timestamp.now()
        }

        st.success("Dataset uploaded successfully!")
        st.dataframe(df)

with tab3:
    st.header("View Datasets")

    if not st.session_state.datasets:
        st.warning("No datasets available. Please add datasets in the Dataset Management page.")
    else:
        for dataset_name, dataset_info in st.session_state.datasets.items():
            with st.expander(f"Dataset: {dataset_name}"):
                st.write(f"Description: {dataset_info['description']}")
                st.write(f"Created at: {dataset_info['created_at']}")
                st.write(f"Shape: {dataset_info['data'].shape}")

                # Display dataset preview
                st.dataframe(dataset_info['data'].head())

                # Add download button
                csv = dataset_info['data'].to_csv(index=False)
                st.download_button(
                    label=f"Download {dataset_name}",
                    data=csv,
                    file_name=f"{dataset_name}.csv",
                    mime="text/csv"
                )
