import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataDriftTable
import json

class MLModel:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = self._get_model()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = None
        self.is_trained = False
        self.shap_explainer = None

    def _get_model(self):
        if self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100)
        elif self.model_type == 'xgboost':
            return XGBClassifier()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess_data(self, df):
        """Preprocess the input data"""
        try:
            # Store feature and target names if not already set
            if self.feature_names is None:
                self.feature_names = df.columns[:-1].tolist()
                self.target_name = df.columns[-1]

            # Create a copy to avoid modifying the original data
            processed_df = df.copy()

            # Handle categorical variables
            for column in processed_df.columns:
                if processed_df[column].dtype == 'object' or processed_df[column].dtype == 'category':
                    if column not in self.label_encoders:
                        self.label_encoders[column] = LabelEncoder()
                        processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column])
                    else:
                        # For prediction, handle unknown categories
                        known_categories = set(self.label_encoders[column].classes_)
                        processed_df[column] = processed_df[column].map(
                            lambda x: x if x in known_categories else -1
                        )
                        processed_df[column] = self.label_encoders[column].transform(processed_df[column])

            # Handle missing values
            processed_df = processed_df.fillna(processed_df.mean())

            # Ensure all columns are numeric
            for column in processed_df.columns:
                if not np.issubdtype(processed_df[column].dtype, np.number):
                    processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce')
                    processed_df[column] = processed_df[column].fillna(processed_df[column].mean())

            return processed_df

        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")

    def train(self, X, y, test_size=0.2):
        """Train the model with the given data"""
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()
            self.target_name = y.name

            # Preprocess the data
            X_processed = self.preprocess_data(X)
            y_processed = y

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=42
            )

            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train the model
            self.model.fit(X_train_scaled, y_train)

            # Calculate metrics
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Calculate SHAP values
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
            except:
                self.shap_explainer = None

            self.is_trained = True

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

        except Exception as e:
            raise ValueError(f"Error training model: {str(e)}")

    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        try:
            # Preprocess the input data
            X_processed = self.preprocess_data(X)

            # Scale the features
            X_scaled = self.scaler.transform(X_processed)

            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)

            return predictions, probabilities

        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")

    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_[0])
            else:
                return None

            return dict(zip(self.feature_names, importance))

        except Exception as e:
            raise ValueError(f"Error getting feature importance: {str(e)}")

    def get_shap_values(self, X):
        """Get SHAP values for feature importance"""
        if not self.is_trained or self.shap_explainer is None:
            return None

        try:
            # Preprocess the input data
            X_processed = self.preprocess_data(X)

            # Scale the features
            X_scaled = self.scaler.transform(X_processed)

            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_scaled)

            # Handle both single and multi-class cases
            if isinstance(shap_values, list):
                return shap_values
            else:
                return [shap_values]

        except Exception as e:
            raise ValueError(f"Error calculating SHAP values: {str(e)}")

    def save_model(self, path):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'model_type': self.model_type
            }
            joblib.dump(model_data, path)
        except Exception as e:
            raise ValueError(f"Error saving model: {str(e)}")

    @classmethod
    def load_model(cls, path):
        """Load a trained model"""
        try:
            model_data = joblib.load(path)
            model = cls(model_data['model_type'])
            model.model = model_data['model']
            model.scaler = model_data['scaler']
            model.label_encoders = model_data['label_encoders']
            model.feature_names = model_data['feature_names']
            model.target_name = model_data['target_name']
            model.is_trained = True
            return model
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

def generate_evidently_report(reference_data, current_data, target_column):
    """Generate an Evidently report for data drift analysis"""
    try:
        # Convert datetime columns to string to avoid Evidently issues
        for df in [reference_data, current_data]:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)

        # Create data drift report
        data_drift_report = Report(metrics=[
            DataDriftTable(),
            DataDriftPreset()
        ])

        # Run the report
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data
        )

        return data_drift_report

    except Exception as e:
        raise ValueError(f"Error generating Evidently report: {str(e)}")

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
