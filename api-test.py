from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
from typing import List

class Features(BaseModel):
    data: List[float]

app = FastAPI()

# Load both model and preprocessor at startup
@app.on_event("startup")
async def load_model():
    global model, preprocessor
    model_dir = Path("models")
    model = joblib.load(model_dir / "model.joblib")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")

@app.get("/")
def demo():
    # Update demo to use realistic sample data
    sample_data = [[0] * len(preprocessor.feature_names_)]  # adjust based on your features
    prediction = model.predict(sample_data)
    return {"demo": float(prediction[0])}

@app.post("/predict")
async def predict(features: Features):
    # Preprocess the input data
    processed_features = preprocessor.transform([features.data])
    predictions = model.predict(processed_features)
    return {"predictions": predictions.tolist()}