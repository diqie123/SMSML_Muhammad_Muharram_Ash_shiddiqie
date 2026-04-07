"""
Capture MLflow screenshots for submission evidence
Author: Muhammad Muharram Ash shiddiqie
"""
from playwright.sync_api import sync_playwright
from pathlib import Path
from urllib.parse import quote
import time

def capture_mlflow_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()

        workspace_root = Path(__file__).resolve().parents[1]
        membangun_model_dir = Path(__file__).resolve().parent
        monitoring_dir = workspace_root / "Monitoring dan Logging"

        prom_evidence_dir = monitoring_dir / "4.bukti monitoring Prometheus"
        graf_evidence_dir = monitoring_dir / "5.bukti monitoring Grafana"
        prom_evidence_dir.mkdir(parents=True, exist_ok=True)
        graf_evidence_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # 1. Screenshot halaman UTAMA MLflow (Experiments list)
        # ============================================================
        page.goto("http://127.0.0.1:8080", wait_until="domcontentloaded")
        time.sleep(3)
        # screenshot halaman utama - daftar experiments
        page.screenshot(
            path=str(membangun_model_dir / "screenshoot_dashboard.jpg"),
            full_page=True, type="jpeg", quality=85
        )
        print("✓ Screenshot dashboard (experiments list) saved")

        # ============================================================
        # 2. Screenshot Artifacts dari run basic autolog
        # ============================================================
        experiment_name = "bank_marketing_classification"
        try:
            page.get_by_text(experiment_name, exact=True).first.click(timeout=10000)
        except Exception:
            # fallback: klik experiment pertama
            page.locator('a[href*="#/experiments/"]').first.click(timeout=10000)
        time.sleep(2)

        # klik pada run yang ada
        try:
            page.locator("table tbody tr").first.locator("a").first.click(timeout=10000)
        except Exception:
            page.locator('a[href*="/runs/"]').first.click(timeout=10000)
        time.sleep(2)

        # klik tab Artifacts
        try:
            page.get_by_text("Artifacts", exact=True).click(timeout=10000)
            time.sleep(2)
        except Exception:
            pass

        page.screenshot(
            path=str(membangun_model_dir / "screenshoot_artifak.jpg"),
            full_page=True, type="jpeg", quality=85
        )
        print("✓ Screenshot artifacts saved")

        # ============================================================
        # 3. Prometheus screenshots
        # ============================================================
        prometheus_base = "http://localhost:19090/graph"
        prometheus_metrics = [
            ("prediction_requests_total", "prediction_requests_total"),
            ("model_accuracy", "model_accuracy"),
            ("prediction_latency_seconds_count", "prediction_latency_seconds_count"),
        ]
        for i, (expr, filename_part) in enumerate(prometheus_metrics, start=1):
            url = f"{prometheus_base}?g0.expr={quote(expr)}&g0.tab=0"
            page.goto(url, wait_until="domcontentloaded")
            time.sleep(2)
            page.screenshot(
                path=str(prom_evidence_dir / f"{i}.monitoring_{filename_part}.png"),
                full_page=True
            )
        print("✓ Prometheus screenshots saved")

        # ============================================================
        # 4. Grafana screenshots (gunakan dashboard bank-dashboard)
        # ============================================================
        grafana_dashboard_url = "http://localhost:13000/d/bank-dashboard"
        page.goto(grafana_dashboard_url, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)
        page.screenshot(
            path=str(graf_evidence_dir / "1.monitoring_dashboard.png"),
            full_page=True
        )
        print("✓ Grafana full dashboard screenshot saved")

        grafana_solo_base = "http://localhost:13000/d-solo/bank-dashboard"
        grafana_panels = [
            (1, "prediction_requests_rate"),
            (2, "model_accuracy"),
            (3, "prediction_latency"),
        ]
        for i, (panel_id, name) in enumerate(grafana_panels, start=2):
            url = f"{grafana_solo_base}?panelId={panel_id}&from=now-1h&to=now&theme=light"
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
            page.screenshot(
                path=str(graf_evidence_dir / f"{i}.monitoring_{name}.png"),
                full_page=True
            )
        print("✓ Grafana panel screenshots saved")

        browser.close()
        print("\n✓ All screenshots captured successfully!")

if __name__ == "__main__":
    capture_mlflow_screenshots()
