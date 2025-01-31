from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from catboost import CatBoostClassifier

# Define model path
MODEL_PATH = "models/catboost_model.cbm"

# Lifespan event handler for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)  # Load model properly
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None  # Prevents crashes if model fails to load
    yield  # No cleanup needed

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
def demo():
    if model is None:
        return {"error": "Model not loaded"}
    
    # Ensure input format is correct for CatBoost
    sample_input = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    prediction = model.predict(sample_input)
    return {"demo": float(prediction[0])}  # Convert NumPy float to standard float

@app.post("/predict")
async def predict(features: List[float]):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Ensure input format is correct
    predictions = model.predict([features])
    return {"predictions": predictions.tolist()}  # Convert to list

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
