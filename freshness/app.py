import os
import io
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from PIL import Image
import uvicorn
from typing import List, Dict, Any
import base64

# Set environment variables to reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize FastAPI app
app = FastAPI(
    title="Freshness Detection API",
    description="API for detecting freshness of fruits and vegetables",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and class names
model = None
class_names = []

@app.on_event("startup")
async def load_model():
    """Load the trained model and class names on startup"""
    global model, class_names
    
    try:
        # Load the model - try multiple model paths
        model_paths = [
            'best_model.h5',
            'final_model.h5',
            'best_efficient_cnn_model.keras',
            os.path.join(os.path.dirname(__file__), 'best_model.h5'),
            os.path.join(os.path.dirname(__file__), 'final_model.h5'),
            os.path.join(os.path.dirname(__file__), 'best_efficient_cnn_model.keras')
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                try:
                    model = tf.keras.models.load_model(model_path)
                    model_loaded = True
                    print(f"✓ Successfully loaded model from {model_path}")
                    break
                except Exception as e:
                    print(f"Failed to load model from {model_path}: {str(e)}")
        
        if not model_loaded:
            raise FileNotFoundError("No compatible model file found. Tried paths: " + ", ".join(model_paths))
        
        # Load class names
        class_names_paths = [
            'class_names.json',
            os.path.join(os.path.dirname(__file__), 'class_names.json')
        ]
        
        class_names_loaded = False
        for class_names_path in class_names_paths:
            if os.path.exists(class_names_path):
                try:
                    with open(class_names_path, 'r') as f:
                        class_names = json.load(f)
                    class_names_loaded = True
                    print(f"✓ Successfully loaded class names from {class_names_path}")
                    break
                except Exception as e:
                    print(f"Failed to load class names from {class_names_path}: {str(e)}")
        
        if not class_names_loaded:
            raise FileNotFoundError("Class names file not found. Tried paths: " + ", ".join(class_names_paths))
        
        print(f"Model initialization complete. Found {len(class_names)} classes.")
        
        # Test prediction with a small tensor to ensure the model works
        test_tensor = tf.zeros((1, 160, 160, 3))
        _ = model.predict(test_tensor, verbose=0)
        print("✓ Model verified with test prediction")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # We'll let the API start, but endpoints will fail until model is loaded

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Open image from memory
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to model input size
        img = img.resize((160, 160))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the index.html file"""
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Freshness Detection API",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Web interface"},
            {"path": "/api", "method": "GET", "description": "This help message"},
            {"path": "/predict", "method": "POST", "description": "Predict freshness from image"},
            {"path": "/health", "method": "GET", "description": "Check API health"},
            {"path": "/classes", "method": "GET", "description": "List available classes"}
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not loaded"}
        )
    return {"status": "healthy", "model": "loaded", "classes": len(class_names)}

@app.get("/classes")
async def get_classes():
    """Return available classes"""
    global class_names
    if not class_names:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Class names not loaded"}
        )
    
    # Group classes by fruit/vegetable type
    fresh_items = [cls for cls in class_names if cls.startswith('fresh')]
    rotten_items = [cls for cls in class_names if cls.startswith('rotten')]
    
    # Extract unique fruit/vegetable names
    fresh_types = [item.replace('fresh', '') for item in fresh_items]
    rotten_types = [item.replace('rotten', '') for item in rotten_items]
    
    return {
        "all_classes": class_names,
        "fresh_items": fresh_items,
        "rotten_items": rotten_items,
        "supported_types": list(set(fresh_types + rotten_types))
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict freshness from uploaded image"""
    global model, class_names
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not class_names:
        raise HTTPException(status_code=503, detail="Class names not loaded")
    
    try:
        # Read image file
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file")
        
        print(f"Received image: {file.filename}, size: {len(image_data)} bytes")
        
        # Preprocess the image
        try:
            img_array, original_img = preprocess_image(image_data)
            print(f"Image preprocessed successfully. Shape: {img_array.shape}")
        except Exception as e:
            print(f"Image preprocessing error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
        
        # Get predictions
        try:
            print("Running model prediction...")
            predictions = model.predict(img_array, verbose=0)
            scores = predictions[0]
            print(f"Prediction completed. Raw scores shape: {scores.shape}")
            
            # Get top 3 predictions
            top_indices = np.argsort(scores)[-3:][::-1]
            
            # Validate indices against class_names
            if max(top_indices) >= len(class_names):
                print(f"WARNING: Prediction index {max(top_indices)} exceeds class_names length {len(class_names)}")
                # Filter to valid indices only
                top_indices = [i for i in top_indices if i < len(class_names)]
                if not top_indices:
                    raise ValueError("No valid class indices found in predictions")
            
            top_classes = [class_names[i] for i in top_indices]
            top_scores = [float(scores[i]) for i in top_indices]
            
            print(f"Top prediction: {top_classes[0]} ({top_scores[0]*100:.2f}%)")
            
            # Format results
            results = []
            for i, (cls, score) in enumerate(zip(top_classes, top_scores)):
                # Parse class name to get type and freshness
                if cls.startswith('fresh'):
                    freshness = "fresh"
                    item_type = cls.replace('fresh', '')
                else:
                    freshness = "rotten"
                    item_type = cls.replace('rotten', '')
                    
                results.append({
                    "rank": i + 1,
                    "class": cls,
                    "score": score,
                    "confidence": f"{score * 100:.2f}%",
                    "freshness": freshness,
                    "item_type": item_type
                })
            
            # Convert image to base64 for optional display in frontend
            try:
                buffered = io.BytesIO()
                original_img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Warning: Failed to encode image as base64: {str(e)}")
                img_base64 = ""
            
            response_data = {
                "filename": file.filename,
                "predictions": results,
                "top_prediction": {
                    "class": top_classes[0],
                    "confidence": f"{top_scores[0] * 100:.2f}%",
                    "freshness_status": "fresh" if top_classes[0].startswith('fresh') else "rotten",
                    "item_type": top_classes[0].replace('fresh', '').replace('rotten', '')
                },
                "image_base64": img_base64
            }
            
            print("Prediction completed successfully")
            return response_data
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in predict endpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)