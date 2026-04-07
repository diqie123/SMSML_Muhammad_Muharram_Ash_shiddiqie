"""
Final Model Serving API using FastAPI
Author: Muhammad Muharram Ash shiddiqie
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os
import uvicorn

# Define data structure for input
class InputData(BaseModel):
    dataframe_split: dict

app = FastAPI(title="Bank Marketing Model Serving API")

# Path ke model terbaru
MODEL_RUN_ID = "23a3319087aa4a77aa0acda89b378650"
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "..", "Membangun_model", "mlruns", "3", MODEL_RUN_ID, "artifacts", "model"))

print(f"Loading model from: {model_path}")
model = mlflow.sklearn.load_model(model_path)

@app.post("/invocations")
async def predict(data: InputData):
    """
    Endpoint for pattern MLflow /invocations
    """
    try:
        # Konversi data ke format pandas untuk model
        df = pd.DataFrame(
            data=data.dataframe_split["data"],
            columns=data.dataframe_split["columns"]
        )
        
        # Lakukan prediksi
        predictions = model.predict(df)
        
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("\nStarting model server on http://127.0.0.1:5002")
    uvicorn.run(app, host="127.0.0.1", port=5002)
