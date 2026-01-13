from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from keras.models import load_model
import tempfile
import os
from typing import Dict, Any, Optional
import uvicorn

# Import the shelf life prediction functions from shell.py
from shelf_life_predictor import predict_shelf_life_api, KINETIC_DATA

app = FastAPI(
    title="Fruit & Vegetable Freshness API",
    description="API for evaluating fruit/vegetable freshness and predicting shelf life",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
try:
    model = load_model('rottenvsfresh98pval.h5')
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

# Pydantic models for request/response
class FreshnessResponse(BaseModel):
    prediction_score: float
    freshness_category: str
    confidence: float
    message: str

class ShelfLifeRequest(BaseModel):
    fruit_name: str
    storage_temperature: float

class ShelfLifeResponse(BaseModel):
    product: str
    storage_temperature: float
    spoilage_metric: str
    activation_energy_kj_mol: float
    degradation_rate: float
    shelf_life_ratio: float
    conclusion: str
    comparison_temperature: float
    # Added: simple characteristic life in days at given temperature
    life_days: float

# Helper functions from evaluate-image.py
def classify_freshness(prediction_score: float) -> Dict[str, Any]:
    """Classify freshness based on prediction score"""
    threshold_fresh = 0.10
    threshold_medium = 0.35
    
    if prediction_score < threshold_fresh:
        category = "FRESH"
        message = "The item is FRESH!"
        confidence = (threshold_fresh - prediction_score) / threshold_fresh
    elif threshold_fresh <= prediction_score < threshold_medium:
        category = "MEDIUM FRESH"
        message = "The item is MEDIUM FRESH"
        confidence = 1 - abs(prediction_score - ((threshold_fresh + threshold_medium) / 2)) / ((threshold_medium - threshold_fresh) / 2)
    else:
        category = "NOT FRESH"
        message = "The item is NOT FRESH"
        confidence = (prediction_score - threshold_medium) / (1 - threshold_medium)
    
    return {
        "category": category,
        "message": message,
        "confidence": min(max(confidence, 0), 1)  # Ensure confidence is between 0 and 1
    }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model prediction"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # Resize and convert color
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize and expand dimensions
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fruit & Vegetable Freshness API",
        "version": "1.0.0",
        "endpoints": {
            "freshness_evaluation": "/evaluate-freshness",
            "shelf_life_prediction": "/predict-shelf-life",
            "available_items": "/available-items"
        }
    }

@app.post("/evaluate-freshness", response_model=FreshnessResponse)
async def evaluate_freshness(file: UploadFile = File(...)):
    """
    Evaluate the freshness of a fruit or vegetable from an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        FreshnessResponse with prediction score, category, and confidence
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check if 'rottenvsfresh98pval.h5' exists.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        prediction_score = float(prediction[0][0])
        
        # Classify freshness
        classification = classify_freshness(prediction_score)
        
        return FreshnessResponse(
            prediction_score=prediction_score,
            freshness_category=classification["category"],
            confidence=classification["confidence"],
            message=classification["message"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-shelf-life", response_model=ShelfLifeResponse)
async def predict_shelf_life_endpoint(request: ShelfLifeRequest):
    """
    Predict shelf life based on fruit/vegetable type and storage temperature.
    
    Args:
        request: ShelfLifeRequest with fruit_name and storage_temperature
    
    Returns:
        ShelfLifeResponse with detailed shelf life analysis
    """
    # Validate fruit name
    fruit_name = request.fruit_name.strip().lower().replace('fresh', '')
    
    if fruit_name not in KINETIC_DATA:
        available_items = list(KINETIC_DATA.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Fruit '{fruit_name}' not found. Available items: {available_items}"
        )
    
    try:
        # Call the shelf life prediction function (only name and temperature)
        result = predict_shelf_life_api(
            fruit_name=fruit_name,
            temp_c=request.storage_temperature,
        )

        return ShelfLifeResponse(
            product=result["product"],
            storage_temperature=result["storage_temperature"],
            spoilage_metric=result["spoilage_metric"],
            activation_energy_kj_mol=result["activation_energy_kj_mol"],
            degradation_rate=result["degradation_rate"],
            shelf_life_ratio=result["shelf_life_ratio"],
            conclusion=result["conclusion"],
            comparison_temperature=result["comparison_temperature"],
            life_days=float(result.get("life_days", 0.0)),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting shelf life: {str(e)}")

@app.get("/available-items")
async def get_available_items():
    """
    Get list of available fruits and vegetables for shelf life prediction.
    
    Returns:
        Dictionary with available items and their spoilage metrics
    """
    items = {}
    for item, data in KINETIC_DATA.items():
        items[item] = {
            "metric": data["metric"],
            "activation_energy_kj_mol": data["Ea"] / 1000
        }
    
    return {
        "available_items": items,
        "total_count": len(items)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "available_items_count": len(KINETIC_DATA)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)