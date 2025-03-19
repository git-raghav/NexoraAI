# ML Project Dashboard

A comprehensive machine learning project dashboard built with Streamlit that allows users to:

-   Train ML models on default datasets (e.g., Iris)
-   Upload and analyze custom datasets
-   Train multiple types of models (Random Forest, Logistic Regression, SVM)
-   Visualize model performance with SHAP values
-   Make predictions using trained models
-   Analyze data drift using Evidently AI

## Features

1. **Dataset Management**

    - Load default datasets (Iris)
    - Upload custom datasets
    - View and manage datasets

2. **Model Training**

    - Train multiple model types
    - Visualize model performance
    - SHAP value analysis
    - Store models in database

3. **Model Prediction**

    - Make predictions using trained models
    - Support for both single predictions and batch predictions
    - Download prediction results

4. **Data Drift Analysis**
    - Compare datasets for drift
    - Analyze feature-level drift
    - Statistical property comparison

## Setup with Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed on your system.

2. Clone this repository:

```bash
git clone <repository-url>
cd <repository-name>
```

3. Build and start the container:

```bash
docker-compose up --build
```

The application will be available at `http://localhost:8501`

To stop the application:

```bash
docker-compose down
```

## Manual Setup (Alternative)

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── database.py            # Database configuration and utilities
├── ml_utils.py           # ML model utilities and helper functions
├── pages/                # Streamlit pages
│   ├── 1_Dataset_Management.py
│   ├── 2_Model_Training.py
│   ├── 3_Model_Prediction.py
│   └── 4_Data_Drift_Analysis.py
├── datasets/             # Directory for stored datasets
├── models/              # Directory for saved models
├── ml_project.db       # SQLite database file
├── Dockerfile           # Dockerfile for building the application
├── docker-compose.yml   # Docker Compose configuration
└── requirements.txt     # Project dependencies
```

## Usage

1. Start by adding datasets in the Dataset Management page
2. Train models on your datasets in the Model Training page
3. Make predictions using your trained models in the Model Prediction page
4. Analyze data drift between datasets in the Data Drift Analysis page

## Dependencies

-   streamlit
-   pandas
-   numpy
-   scikit-learn
-   shap
-   evidently
-   python-dotenv
-   plotly
-   joblib
-   matplotlib

## Notes

-   The project uses SQLite to store dataset and model metadata
-   Models and datasets are stored on the filesystem
-   SHAP values are used for model interpretability
-   Evidently AI is used for data drift analysis
-   Docker volumes are used to persist data and models between container restarts

Happy Coding!! 
