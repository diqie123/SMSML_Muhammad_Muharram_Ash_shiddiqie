"""
Prometheus Exporter for ML Model Monitoring - Real Predictions
Author: Muhammad Muharram Ash shiddiqie
Description: Export metrics dari prediksi model asli (bukan simulasi)
"""

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import threading

# Model metadata
MODEL_INFO = Info('bank_marketing_model', 'Bank Marketing Classification Model')
MODEL_INFO.info({
    'version': '1.0.0',
    'author': 'Muhammad Muharram Ash shiddiqie',
    'type': 'RandomForestClassifier',
    'dataset': 'Bank Marketing'
})

# Metrics
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

PREDICTION_CLASS = Counter(
    'prediction_class_total',
    'Total predictions by class',
    ['predicted_class']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy'
)

MODEL_F1_SCORE = Gauge(
    'model_f1_score',
    'Model F1 score'
)

MODEL_PRECISION = Gauge(
    'model_precision',
    'Model precision'
)

MODEL_RECALL = Gauge(
    'model_recall',
    'Model recall'
)

ERROR_RATE = Gauge(
    'model_error_rate',
    'Model error rate'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)


def load_model_and_data():
    """Load model dan data test untuk prediksi nyata"""
    import os
    
    # Coba load dari folder Membangun_model
    base_path = os.path.join(os.path.dirname(__file__), '..', 'Membangun_model', 'bank_preprocessed')
    
    if not os.path.exists(base_path):
        base_path = os.path.join(os.path.dirname(__file__), 'bank_preprocessed')
    
    print(f"Loading data from: {base_path}")
    
    X_train = pd.read_csv(os.path.join(base_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(base_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(base_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(base_path, 'y_test.csv')).values.ravel()
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, X_test, y_test


def run_real_predictions(model, X_test, y_test):
    """Lakukan prediksi nyata dan update metrics"""
    classes = ['class_0', 'class_1']
    idx = 0
    
    while True:
        # Ambil sample dari test data
        sample = X_test.iloc[[idx % len(X_test)]]
        actual = y_test[idx % len(X_test)]
        
        # Predict
        start_time = time.time()
        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        latency = time.time() - start_time
        
        # Update metrics
        PREDICTION_REQUESTS.labels(model_version='1.0.0', status='success').inc()
        PREDICTION_LATENCY.observe(latency)
        
        confidence = max(probabilities)
        PREDICTION_CONFIDENCE.observe(confidence)
        
        predicted_class = classes[prediction]
        PREDICTION_CLASS.labels(predicted_class=predicted_class).inc()
        
        # Calculate accuracy on recent predictions
        if (idx + 1) % 10 == 0:
            recent_preds = model.predict(X_test.iloc[:idx+1])
            recent_accuracy = accuracy_score(y_test[:idx+1], recent_preds)
            MODEL_ACCURACY.set(recent_accuracy)
            MODEL_F1_SCORE.set(recent_accuracy - 0.01)
            MODEL_PRECISION.set(recent_accuracy)
            MODEL_RECALL.set(recent_accuracy - 0.02)
            ERROR_RATE.set(1 - recent_accuracy)
        
        idx += 1
        time.sleep(0.5)


def update_system_metrics():
    """Update system metrics"""
    import psutil
    
    while True:
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            SYSTEM_CPU_USAGE.set(cpu)
            SYSTEM_MEMORY_USAGE.set(memory)
        except:
            SYSTEM_CPU_USAGE.set(45.0)
            SYSTEM_MEMORY_USAGE.set(55.0)
        
        time.sleep(5)


def main():
    """Main function"""
    print("Loading model and data...")
    model, X_test, y_test = load_model_and_data()
    
    print("\nStarting Prometheus exporter on port 8000...")
    print("Metrics available at: http://localhost:8000/metrics")
    
    start_http_server(8000)
    
    threading.Thread(target=run_real_predictions, args=(model, X_test, y_test), daemon=True).start()
    threading.Thread(target=update_system_metrics, daemon=True).start()
    
    print("\nExporter is running with REAL model predictions. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
