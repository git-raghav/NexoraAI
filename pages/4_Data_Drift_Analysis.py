import streamlit as st
import pandas as pd
from database import get_database_connection
from ml_utils import generate_evidently_report
import json

st.title("Data Drift Analysis")

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
    # Reference dataset selection
    st.subheader("Select Reference Dataset")
    reference_dataset = st.selectbox(
        "Reference Dataset",
        datasets,
        format_func=lambda x: x[1],
        key="reference"
    )

    # Current dataset selection
    st.subheader("Select Current Dataset")
    current_dataset = st.selectbox(
        "Current Dataset",
        datasets,
        format_func=lambda x: x[1],
        key="current"
    )

    if st.button("Analyze Data Drift"):
        # Load datasets
        reference_df = pd.read_csv(f"datasets/{reference_dataset[1]}.csv")
        current_df = pd.read_csv(f"datasets/{current_dataset[1]}.csv")

        # Generate Evidently report
        target_column = reference_df.columns[-1]  # Assuming last column is target
        report = generate_evidently_report(reference_df, current_df, target_column)

        # Display report sections
        st.subheader("Data Drift Analysis Results")

        # Convert report to JSON for easier parsing
        report_json = json.loads(report.json())

        # Display Data Drift results
        st.write("### Data Drift Analysis")
        data_drift_metrics = report_json['metrics'][0]

        # Overall drift
        st.write(f"Data Drift Detected: {data_drift_metrics['result']['data_drift_detected']}")
        st.write(f"Share of Drifted Features: {data_drift_metrics['result']['share_of_drifted_features']:.2%}")

        # Feature-level drift
        st.write("\n### Feature-level Drift Analysis")
        for feature, drift_score in data_drift_metrics['result']['drift_by_columns'].items():
            if feature != target_column:
                st.write(f"- {feature}: {'Drift detected' if drift_score['drift_detected'] else 'No drift'}")
                st.write(f"  p-value: {drift_score['p_value']:.4f}")

        # Target Drift results
        st.write("\n### Target Drift Analysis")
        target_drift_metrics = report_json['metrics'][1]
        st.write(f"Target Drift Detected: {target_drift_metrics['result']['target_drift']['drift_detected']}")
        st.write(f"Target Drift p-value: {target_drift_metrics['result']['target_drift']['p_value']:.4f}")

        # Statistical properties
        st.write("\n### Statistical Properties")
        for feature in reference_df.columns:
            if feature != target_column:
                st.write(f"\n#### {feature}")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Reference Dataset:")
                    st.write(f"- Mean: {reference_df[feature].mean():.2f}")
                    st.write(f"- Std: {reference_df[feature].std():.2f}")

                with col2:
                    st.write("Current Dataset:")
                    st.write(f"- Mean: {current_df[feature].mean():.2f}")
                    st.write(f"- Std: {current_df[feature].std():.2f}")
