"""
Model Training with MLflow Autolog - Basic Level
Author: Muhammad Muharram Ash shiddiqie
Description: Training model dengan MLflow autolog (BASIC criteria)
             - Menggunakan mlflow.sklearn.autolog() untuk mencatat metrik dan parameter otomatis
             - Model disimpan lokal di mlruns/
             - Jalankan: python modelling.py
             - View MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ============================================================
# BASIC CRITERIA: MLflow Autolog (otomatis catat params & metrics)
# ============================================================

# Set MLflow tracking URI (lokal SQLite)
script_dir = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"sqlite:///{script_dir}/mlflow.db")
mlflow.set_experiment("bank_marketing_classification")

# Enable autolog SEBELUM training - ini yang menentukan kriteria BASIC
mlflow.sklearn.autolog(log_models=True)


def load_data(data_dir=None):
    """Load preprocessed data"""
    if data_dir is None:
        data_dir = os.path.join(script_dir, "bank_preprocessed")
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    """Train model dengan MLflow autolog untuk Basic criteria"""
    
    with mlflow.start_run(run_name="basic_rf_model_autolog"):
        # ============================================================
        # AUTOLOG akan otomatis mencatat:
        # - Parameters: n_estimators, max_depth, dll
        # - Metrics: accuracy, precision, recall, f1
        # - Artifacts: model/, estimator.html
        # ============================================================
        
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Log additional info manually if needed
        mlflow.log_param("model_type", "RandomForest")
        
        print(f"\n{'='*50}")
        print(f"Model trained successfully!")
        print(f"Experiment: bank_marketing_classification")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{report}")
        print(f"\nArtifacts saved to MLflow (local: mlruns/)")
        print(f"View MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        
        return model, accuracy


def main():
    """Main function"""
    print("="*50)
    print("Bank Marketing - Model Training (Basic - Autolog)")
    print("="*50)
    
    print("\nLoading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")
    
    print("\nTraining model with MLflow autolog...")
    model, accuracy = train_model(X_train, X_test, y_train, y_test)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
