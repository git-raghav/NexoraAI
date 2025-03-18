import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import json

class MLModel:
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def get_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42)
        elif self.model_type == "svm":
            return SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def preprocess_data(self, X):
        # Store feature names
        self.feature_names = X.columns.tolist()

        # Create a copy of the data
        X_processed = X.copy()

        # Process each column
        for column in X.columns:
            if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    X_processed[column] = self.label_encoders[column].fit_transform(X[column])
                else:
                    X_processed[column] = self.label_encoders[column].transform(X[column])

        return X_processed

    def train(self, X, y, test_size=0.2):
        # Preprocess features
        X_processed = self.preprocess_data(X)

        # Encode target if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.label_encoders['target'] = LabelEncoder()
            y = self.label_encoders['target'].fit_transform(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)

        # Train the model
        self.model = self.get_model()
        self.model.fit(X_train_scaled, y_train)

        # Get predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics
        }

    def get_shap_values(self, X):
        X_processed = self.preprocess_data(X)
        X_scaled = self.scaler.transform(X_processed)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        if self.model_type in ["random_forest", "logistic_regression"]:
            explainer = shap.TreeExplainer(self.model) if self.model_type == "random_forest" else shap.LinearExplainer(self.model, X_scaled_df)
            shap_values = explainer.shap_values(X_scaled_df)
            return shap_values
        return None

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, path)

    @staticmethod
    def load_model(path):
        model_data = joblib.load(path)
        ml_model = MLModel(model_type=model_data['model_type'])
        ml_model.model = model_data['model']
        ml_model.scaler = model_data['scaler']
        ml_model.label_encoders = model_data['label_encoders']
        ml_model.feature_names = model_data['feature_names']
        return ml_model

def generate_evidently_report(reference_data, current_data, target_column):
    # Create column mapping
    column_mapping = {
        'target': target_column,
        'numerical_features': reference_data.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_features': reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
    }

    # Remove target from features lists
    if target_column in column_mapping['numerical_features']:
        column_mapping['numerical_features'].remove(target_column)
    if target_column in column_mapping['categorical_features']:
        column_mapping['categorical_features'].remove(target_column)

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    return report

def load_default_dataset(dataset_name):
    if dataset_name == "iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def process_uploaded_dataset(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[:, :-1], df.iloc[:, -1]  # Assumes last column is target
