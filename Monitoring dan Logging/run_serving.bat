@echo off
echo Starting MLflow Model Serving...
echo Please take a screenshot of this window as proof of model serving.
echo.
mlflow models serve -m "models:/bank_marketing_model/1" --port 5002 --no-conda
pause
