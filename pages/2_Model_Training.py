import streamlit as st
import pandas as pd
import numpy as np
from ml_utils import MLModel
import json

st.title("Model Training")

# Get available datasets from session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

if not st.session_state.datasets:
    st.warning("No datasets available. Please add datasets in the Dataset Management page.")
else:
    # Dataset selection
    dataset_name = st.selectbox(
        "Select Dataset",
        list(st.session_state.datasets.keys())
    )

    # Model configuration
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Select Model Type",
        ["random_forest", "logistic_regression", "svm"]
    )

    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)

    if st.button("Train Model"):
        # Get dataset from session state
        dataset_info = st.session_state.datasets[dataset_name]
        df = dataset_info['data']

        # Split features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Create and train model
        model = MLModel(model_type=model_type)
        results = model.train(X, y, test_size=test_size)

        # Save model to session state
        if 'models' not in st.session_state:
            st.session_state.models = {}

        model_name = f"{dataset_name}_{model_type}"
        st.session_state.models[model_name] = {
            'model': model,
            'results': results,
            'dataset_name': dataset_name,
            'model_type': model_type
        }

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
