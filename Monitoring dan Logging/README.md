# Monitoring dan Logging - Bank Marketing Model

Sistem monitoring lengkap untuk model Bank Marketing Classification menggunakan Prometheus dan Grafana.

## 📋 Deskripsi

Project ini berisi implementasi monitoring dan serving untuk model Bank Marketing:

- **Model Serving (API)**: FastAPI server yang melayani prediksi real-time (Kriteria 4)
- **Real-time Metrics**: Tracking prediction requests, latency, accuracy, dan system resources via Prometheus
- **Grafana Dashboard**: Visualisasi metrics interaktif dengan nama **Muhammad_Muharram_Ash_shiddiqie**
- **Inference Client**: Script simulasi pengiriman data ke model server

## 📁 Struktur File

```
Monitoring dan Logging/
├── 1.bukti_serving/          # Screenshot bukti model telah di-serve (FastAPI)
│   └── screenshoot_serving.jpg
├── 2.prometheus.yml          # Konfigurasi Prometheus untuk scraping metrics
├── 3.prometheus_exporter.py  # Exporter metrics (latensi, akurasi, dll)
├── 4.bukti monitoring Prometheus/  # Screenshot metrics di Prometheus
├── 5.bukti monitoring Grafana/     # Screenshot dashboard Grafana (Username valid)
├── 6.bukti alerting Grafana/       # Bukti konfigurasi alerting
├── 7.inference.py            # Client untuk simulasi inferensi
├── 8.serving_api.py          # Backend API Server (FastAPI) untuk serving model
├── docker-compose.yml        # Docker setup (Prometheus + Grafana)
├── grafana_dashboard.json    # Ekspor konfigurasi dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # Dokumentasi ini
```

## 🚀 Cara Menjalankan

### Prerequisites
- Docker & Docker Compose
- Python 3.12+
- Library: `fastapi`, `uvicorn`, `mlflow`, `prometheus-client`, `pandas`, `scikit-learn`

---

### 1. Jalankan Model Serving API (Kriteria 4)
Jalankan backend API yang akan melayani prediksi:
```powershell
python 8.serving_api.py
```
Akses health check: `http://127.0.0.1:5002/health`

---

### 2. Jalankan Infrastruktur Monitoring (Docker)
Jalankan Prometheus dan Grafana:
```powershell
docker-compose up -d
```
- Prometheus: `http://localhost:19090`
- Grafana: `http://localhost:13000` (admin/admin)

---

### 3. Jalankan Metrics Exporter
Jalankan script untuk mengumpulkan dan mengekspos metrics ke Prometheus:
```powershell
python 3.prometheus_exporter.py
```
Endpoint Metrics: `http://localhost:8000/metrics`

---

### 4. Simulasi Inferensi
Kirim request ke Model Server untuk melihat pergerakan metrics:
```powershell
python 7.inference.py
```

---

## 📊 Metrics Terpantau
- `prediction_requests_total`: Total request yang masuk.
- `prediction_latency_seconds`: Latensi pemrosesan model.
- `model_accuracy`: Akurasi model secara real-time.
- `system_cpu_usage_percent`: Beban CPU server.
- `system_memory_usage_percent`: Penggunaan RAM server.

---

## 📞 Kontak
- **Author**: Muhammad Muharram Ash shiddiqie
- **GitHub**: [diqie123](https://github.com/diqie123)

---

**Happy Monitoring! 🚀**
