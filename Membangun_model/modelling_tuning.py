"""
Model Training with Hyperparameter Tuning - Robust Version
Author: Muhammad Muharram Ash shiddiqie
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import sys

# Set MLflow tracking URI (SQLite database)
script_dir = os.path.dirname(os.path.abspath(__file__))
tracking_uri = f"sqlite:///{script_dir}/mlflow.db"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("bank_marketing_classification_tuning")


def load_data():
    """Load preprocessed bank marketing data"""
    data_dir = os.path.join(script_dir, "bank_preprocessed")
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, X_test, y_train, y_test


def main():
    """Main function"""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    }
    
    run = mlflow.start_run(run_name="tuned_rf_model")
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    
    try:
        # Log parameter grid
        mlflow.log_param("param_grid", str(param_grid))
        
        # Grid search
        print("Performing hyperparameter tuning...")
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", float(grid_search.best_score_))
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_std", float(cv_scores.std()))
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Log metrics
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1_score", float(f1))
        
        # 1. Manual Log: Confusion Matrix Plot
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax, cmap='Blues')
        plt.title('Confusion Matrix - Tuned RF Model')
        conf_matrix_path = os.path.join(script_dir, "confusion_matrix.png")
        plt.savefig(conf_matrix_path)
        plt.close()
        mlflow.log_artifact(conf_matrix_path)
        os.remove(conf_matrix_path)
        
        # 2. Manual Log: Feature Importance Plot
        import seaborn as sns
        
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Top 15 Feature Importances')
        feat_imp_path = os.path.join(script_dir, "feature_importance.png")
        plt.savefig(feat_imp_path)
        plt.close()
        mlflow.log_artifact(feat_imp_path)
        os.remove(feat_imp_path)
        
        # 3. Manual Log: Classification Report
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred)
        report_path = os.path.join(script_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)
        
        # Save model params as artifact
        model_params = {
            "model_type": "RandomForestClassifier",
            "best_params": best_params,
            "cv_scores": {"mean": float(cv_scores.mean()), "std": float(cv_scores.std())},
            "test_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        }
        
        params_path = os.path.join(script_dir, "model_params.json")
        with open(params_path, "w") as f:
            json.dump(model_params, f, indent=2)
        mlflow.log_artifact(params_path)
        os.remove(params_path)
        
        # Log model
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(best_model, "model")
        
        # Explicitly end run with FINISHED status
        mlflow.end_run(status="FINISHED")
        
        print(f"\nModel trained successfully!")
        print(f"Run ID: {run_id}")
        print(f"Best Parameters: {best_params}")
        print(f"CV Score: {grid_search.best_score_:.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        mlflow.end_run(status="FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
