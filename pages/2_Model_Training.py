import streamlit as st
import pandas as pd
import numpy as np
from database import get_database_connection
from ml_utils import MLModel, process_uploaded_dataset
import os
import json

st.title("Model Training")

# Get available datasets
conn = get_database_connection()
datasets = []
if conn:
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM datasets")
    datasets = cur.fetchall()
    cur.close()
    conn.close()

if not datasets:
    st.warning("No datasets available. Please add datasets in the Dataset Management page.")
else:
    # Dataset selection
    dataset_id, dataset_name = st.selectbox(
        "Select Dataset",
        datasets,
        format_func=lambda x: x[1]
    )

    # Model configuration
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Select Model Type",
        ["random_forest", "logistic_regression", "svm"]
    )

    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)

    if st.button("Train Model"):
        # Load dataset
        df = pd.read_csv(f"datasets/{dataset_name}.csv")
        X, y = process_uploaded_dataset(f"datasets/{dataset_name}.csv")

        # Create and train model
        model = MLModel(model_type=model_type)
        results = model.train(X, y, test_size=test_size)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{dataset_name}_{model_type}.joblib"
        model.save_model(model_path)

        # Save to database
        conn = get_database_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO models (name, dataset_id, model_type, model_path, metrics) VALUES (%s, %s, %s, %s, %s)",
                (f"{dataset_name}_{model_type}", dataset_id, model_type, model_path, json.dumps(results['metrics']))
            )
            conn.commit()
            cur.close()
            conn.close()

        # Display results
        st.success("Model trained successfully!")

        # Display metrics
        st.subheader("Model Performance")
        st.write("Accuracy:", results['metrics']['accuracy'])

        # Display classification report
        st.write("Classification Report:")
        st.json(results['metrics']['classification_report'])

        # Generate and display SHAP values
        st.subheader("SHAP Values")
        shap_values = model.get_shap_values(X)
        if shap_values is not None:
            import shap

            # Create SHAP summary plot
            st.write("SHAP Summary Plot")
            fig = shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig)

            # Feature importance based on SHAP values
            st.write("Feature Importance")
            feature_importance = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            st.dataframe(importance_df)
