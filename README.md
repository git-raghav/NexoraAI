# ML Project Management System

A Streamlit-based application for managing machine learning projects, including dataset management, model training, prediction, and data drift analysis.

## Features

-   Dataset Management
    -   Upload and manage datasets
    -   View dataset statistics and visualizations
    -   Support for CSV files
-   Model Training
    -   Train multiple ML models
    -   Model performance comparison
    -   Feature importance analysis
-   Model Prediction
    -   Make predictions on new data
    -   Support for both manual input and CSV upload
    -   SHAP value visualization
-   Data Drift Analysis
    -   Compare datasets for drift
    -   Statistical analysis
    -   Feature-level drift detection

## Project Structure

```
.
├── README.md
├── requirements.txt
├── ml_utils.py
├── database.py
├── ml_project.db
├── datasets/
│   └── .gitkeep
└── pages/
    ├── 1_Dataset_Management.py
    ├── 2_Model_Training.py
    ├── 3_Model_Prediction.py
    └── 4_Data_Drift_Analysis.py
```

## Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

1. Start the Streamlit app:

```bash
streamlit run Home.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Deployment on Streamlit Cloud

1. Push your code to a GitHub repository

2. Go to [Streamlit Cloud](https://share.streamlit.io/)

3. Click "New app" and select your repository

4. Set the following configuration:

    - Main file path: `Home.py`
    - Python version: 3.8 or higher
    - Add the following secrets in Streamlit Cloud:
        ```
        [general]
        enableTelemetry = false
        ```

5. Click "Deploy!"

## Usage

1. **Dataset Management**

    - Upload your datasets using the Dataset Management page
    - View dataset statistics and visualizations
    - Datasets are stored in the `datasets` directory

2. **Model Training**

    - Select a dataset and model type
    - Configure model parameters
    - Train and evaluate models
    - View model performance metrics and feature importance

3. **Model Prediction**

    - Select a trained model
    - Input new data manually or upload a CSV file
    - Get predictions and probability scores
    - View SHAP values for feature importance

4. **Data Drift Analysis**
    - Select reference and current datasets
    - Analyze data drift between datasets
    - View statistical properties and drift metrics

## Dependencies

-   streamlit
-   pandas
-   numpy
-   scikit-learn
-   shap
-   evidently
-   plotly
-   seaborn

## Notes

-   The application uses SQLite for storing dataset and model metadata
-   Datasets are stored in the `datasets` directory
-   The SQLite database file (`ml_project.db`) is created automatically on first run
-   All data is persisted between sessions using Streamlit's session state and SQLite
-   SHAP values are used for model interpretability
-   Evidently AI is used for data drift analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Happy Coding!!
