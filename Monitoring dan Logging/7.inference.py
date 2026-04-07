import pandas as pd
import requests
import json
import os
import time

def main():
    print("Loading test data...")
    base_path = os.path.join(os.path.dirname(__file__), '..', 'Membangun_model', 'bank_preprocessed')
    if not os.path.exists(base_path):
        base_path = os.path.join(os.path.dirname(__file__), 'bank_preprocessed')

    # Load data preparation
    x_test = pd.read_csv(os.path.join(base_path, 'X_test.csv'))
    
    # Ambil 3 sampel teratas / acak
    sample_data = x_test.head(3)
    
    # URL model API yang diserving (MLflow)
    url = "http://127.0.0.1:5002/invocations"
    
    # Format split JSON untuk requirements MLflow backend serve
    data = {
        "dataframe_split": {
            "columns": list(sample_data.columns),
            "data": sample_data.values.tolist()
        }
    }

    headers = {"Content-Type": "application/json"}
    
    print(f"\nMengirim request ke URL Serving Model: {url}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data, headers=headers)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"Request berhasil! Waktu respons: {latency:.4f} detik")
            print(f"Hasil Prediksi Backend Model Serving:\n{json.dumps(predictions, indent=2)}")
        else:
            print(f"Gagal melakukan inferensi. Status code: {response.status_code}")
            print(f"Error Message API: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"Gagal/Error: Tidak dapat terhubung ke model di {url}")
        print("Pastikan MLflow Model Serving backend sudah dijalankan di background (port 5002).")
        print("Contoh command: mlflow models serve -m \"models:/bank_marketing_model/1\" --port 5002 --no-conda")

if __name__ == "__main__":
    main()
