# Freshness Detection System

A computer vision-based system to detect whether fruits and vegetables are fresh or rotten using a CNN model.

## Overview

This system uses a trained convolutional neural network (CNN) to classify images of fruits and vegetables as either fresh or rotten. It includes:

- A web interface for uploading and analyzing images
- A RESTful API for integration with other systems
- Pre-trained models for accurate classification
- Support for multiple types of produce

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Web browser
- Internet connection (for initial setup only)

### Getting Started

1. Make sure you have all the model files in place:
   - `best_model.h5` or `final_model.h5` (trained CNN model)
   - `class_names.json` (class labels for prediction)

2. Start the server:
   ```
   start_server.bat
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Using the Web Interface

1. Click the "Choose File" button to select an image
2. Click "Analyze" to process the image
3. View the prediction results, including:
   - Classification (Fresh or Rotten)
   - Confidence score
   - Type of produce

## Troubleshooting

If you encounter issues:

1. Check if the server is running correctly
2. Verify that the model files are present
3. Try the diagnostic tool: `start_server.bat check`
4. Check the detailed troubleshooting guide in `TROUBLESHOOTING.md`

## API Endpoints

- `GET /` - Web interface
- `GET /api` - API documentation
- `POST /predict` - Submit an image for analysis
- `GET /health` - Check system status
- `GET /classes` - List available classes

## Example API Usage

```python
import requests

# Send an image to the API
with open('apple.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

# Parse the result
result = response.json()
print(f"Prediction: {result['top_prediction']['class']}")
print(f"Confidence: {result['top_prediction']['confidence']}")
```

## Models

The system includes several pre-trained models:

- `best_model.h5` - Optimized CNN model (default)
- `final_model.h5` - Alternative model
- `best_efficient_cnn_model.keras` - Efficient CNN variant

## Supported Produce Types

- Apple
- Banana
- Bell Pepper
- Bitter Gourd
- Capsicum
- Carrot
- Cucumber
- Mango
- Okra
- Orange
- Potato
- Strawberry
- Tomato

## License

This project is licensed for educational and research purposes only.

## Acknowledgements

- TensorFlow and Keras for the deep learning framework
- FastAPI for the web API framework
- The dataset contributors for providing training data