# Fruit & Vegetable Freshness API

A FastAPI-based backend service that provides two main functionalities:

1. **Freshness Evaluation**: Upload an image to evaluate the freshness of fruits and vegetables
2. **Shelf Life Prediction**: Predict shelf life based on storage temperature using Arrhenius kinetics

## 🚀 Features

### Freshness Evaluation Endpoint (`/evaluate-freshness`)
- Upload fruit/vegetable images for freshness classification
- Returns prediction score, freshness category (FRESH, MEDIUM FRESH, NOT FRESH), and confidence level
- Uses a pre-trained MobileNetV2-based model (`rottenvsfresh98pval.h5`)

### Shelf Life Prediction Endpoint (`/predict-shelf-life`)
- Predict shelf life based on product type and storage temperature
- Uses Arrhenius equation with hardcoded kinetic parameters
- Supports 13 different fruits and vegetables
- Compares shelf life relative to 5°C storage temperature

### Additional Endpoints
- `/available-items`: Get list of supported fruits/vegetables
- `/health`: API health check
- `/`: API information and documentation

## 📋 Supported Items for Shelf Life Prediction

- Apple (firmness loss)
- Banana (softening/ripening)
- Bell Pepper (color degradation)
- Bitter Gourd (estimated weight loss)
- Capsicum (color degradation)
- Carrot (vitamin C loss)
- Cucumber (firmness loss)
- Mango (firmness/ripening)
- Okra (estimated respiration rate)
- Orange (vitamin C loss)
- Potato (texture/ascorbic acid loss)
- Strawberry (anthocyanin color loss)
- Tomato (softening/respiration)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- The trained model file `rottenvsfresh98pval.h5` (should be in the project directory)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Model File
Make sure `rottenvsfresh98pval.h5` is in the project directory. If missing, the freshness evaluation endpoint will return an error.

### Step 3: Start the Server
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📖 Usage

### Option 1: Web Interface
Open `frontend.html` in your web browser for a user-friendly interface to test both endpoints.

### Option 2: Test Script
Run the test client to verify all endpoints:
```bash
python test_api.py
```

### Option 3: Direct API Calls

#### Freshness Evaluation
```python
import requests

# Upload an image for freshness evaluation
with open('your_image.jpg', 'rb') as image_file:
    files = {'file': image_file}
    response = requests.post('http://localhost:8000/evaluate-freshness', files=files)
    result = response.json()
    print(result)
```

#### Shelf Life Prediction
```python
import requests

# Predict shelf life
data = {
    "fruit_name": "apple",
    "storage_temperature": 25.0
}
response = requests.post('http://localhost:8000/predict-shelf-life', json=data)
result = response.json()
print(result)
```

## 📊 API Response Examples

### Freshness Evaluation Response
```json
{
    "prediction_score": 0.0234,
    "freshness_category": "FRESH",
    "confidence": 0.766,
    "message": "The item is FRESH!"
}
```

### Shelf Life Prediction Response
```json
{
    "product": "Apple",
    "storage_temperature": 25.0,
    "spoilage_metric": "Firmness loss",
    "activation_energy_kj_mol": 70.0,
    "degradation_rate": 0.00012,
    "shelf_life_ratio": 0.15,
    "conclusion": "Spoilage is 6.67 times FASTER than at 5°C.",
    "comparison_temperature": 5.0
}
```

## 🔧 Configuration

### Model Path
The API looks for the model file `rottenvsfresh98pval.h5` in the current directory. You can modify the model path in `main.py` if needed.

### CORS Settings
The API is configured to allow all origins for development. Modify the CORS middleware in `main.py` for production use.

### Server Settings
Default server settings:
- Host: `0.0.0.0`
- Port: `8000`

Modify these in the `uvicorn.run()` call in `main.py`.

## 📁 Project Structure
```
fruit-veg-freshness-ai-main/
├── main.py                 # FastAPI application
├── shelf_life_predictor.py # Shelf life prediction logic
├── evaluate-image.py       # Original image evaluation script
├── shell.py               # Original shell-based predictor
├── requirements.txt       # Python dependencies
├── test_api.py           # API test client
├── frontend.html         # Web interface
├── rottenvsfresh98pval.h5 # Trained model (required)
└── README_API.md         # This file
```

## 🚨 Troubleshooting

### Model Not Loading
- Ensure `rottenvsfresh98pval.h5` exists in the project directory
- Check that TensorFlow/Keras is properly installed
- Verify the model file is not corrupted

### Image Upload Issues
- Supported formats: JPG, PNG, GIF, etc.
- Maximum file size depends on FastAPI settings
- Ensure proper Content-Type headers

### Shelf Life Prediction Errors
- Check that the fruit name is in the supported list
- Verify temperature is a valid number
- Use lowercase fruit names (API will handle case conversion)

## 📚 API Documentation

When the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Original Scripts

The API maintains backward compatibility with the original scripts:
- `evaluate-image.py`: Can still be used standalone
- `shell.py`: Console-based shelf life predictor

## 📝 License

This project maintains the same license as the original codebase.