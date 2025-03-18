import streamlit as st
import pandas as pd
import numpy as np
from ml_utils import MLModel

st.title("Model Prediction")

# Get available models from session state
if 'models' not in st.session_state:
    st.session_state.models = {}

if not st.session_state.models:
    st.warning("No trained models available. Please train models in the Model Training page.")
else:
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        list(st.session_state.models.keys())
    )

    # Get model from session state
    model_info = st.session_state.models[model_name]
    model = model_info['model']

    # Input method selection
    input_method = st.radio(
        "Select input method",
        ["Manual Input", "CSV Upload"]
    )

    if input_method == "Manual Input":
        # Get the dataset structure
        dataset_info = st.session_state.datasets[model_info['dataset_name']]
        df = dataset_info['data']
        feature_names = df.columns[:-1]  # Exclude target column

        # Create input fields for each feature
        input_data = {}
        for feature in feature_names:
            if df[feature].dtype in ['int64', 'float64']:
                input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
            else:
                unique_values = df[feature].unique()
                input_data[feature] = st.selectbox(f"Select value for {feature}", unique_values)

        if st.button("Make Prediction"):
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = model.model.predict(model.scaler.transform(input_df))
            prediction_proba = model.model.predict_proba(model.scaler.transform(input_df))

            # Display results
            st.success(f"Prediction: {prediction[0]}")
            st.write("Prediction Probabilities:")
            proba_df = pd.DataFrame(
                prediction_proba,
                columns=[f"Class {i}" for i in range(len(prediction_proba[0]))]
            )
            st.dataframe(proba_df)

            # Display SHAP values for this prediction
            shap_values = model.get_shap_values(input_df)
            if shap_values is not None:
                st.subheader("Feature Importance for this Prediction")
                if isinstance(shap_values, list):
                    # For multi-class problems
                    shap_values = shap_values[prediction[0]]
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': np.abs(shap_values[0])
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance_df)

    else:  # CSV Upload
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)

            if st.button("Make Predictions"):
                # Make predictions
                predictions = model.model.predict(model.scaler.transform(input_df))
                predictions_proba = model.model.predict_proba(model.scaler.transform(input_df))

                # Add predictions to the input DataFrame
                results_df = input_df.copy()
                results_df['Prediction'] = predictions

                # Add probability columns
                for i in range(predictions_proba.shape[1]):
                    results_df[f'Probability_Class_{i}'] = predictions_proba[:, i]

                # Display results
                st.success("Predictions made successfully!")
                st.dataframe(results_df)

                # Download button for results
                st.download_button(
                    label="Download predictions as CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv'
                )
