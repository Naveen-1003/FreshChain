# Freshness Detection Troubleshooting Guide

This guide will help you resolve the "NetworkError when attempting to fetch resource" error and other issues with the CNN model.

## Quick Solutions

### 1. Check the Model First

Run the model diagnostic tool to verify everything is set up correctly:

```
start_server.bat check
```

This will verify that:
- The model files exist
- The class names file is valid
- The model can be loaded successfully
- The prediction function works properly

### 2. Start the Server in Debug Mode

If the diagnostic passes but you still have issues, try starting the server with extra logging:

```
start_server.bat debug
```

This will enable detailed logs from TensorFlow and the API server.

### 3. Common Issues and Solutions

#### Model Loading Failures

- **Problem**: "No model file found" or "Failed to load model"
- **Solution**: Make sure `best_model.h5`, `final_model.h5` or `best_efficient_cnn_model.keras` exists in the root directory.

#### Network Errors in the Browser

- **Problem**: "NetworkError when attempting to fetch resource" in the browser console
- **Solution**: 
  - Check the server logs for Python errors
  - Make sure the image isn't too large (should be under 10MB)
  - Try with a different image
  - Verify your browser can connect to http://localhost:8000

#### 500 Server Errors

- **Problem**: Server returns HTTP 500 errors
- **Solution**: 
  - Check server logs for detailed error messages
  - Make sure the model is compatible with your TensorFlow version
  - Check if the class names in `class_names.json` match what the model expects

## Troubleshooting Steps

1. **Verify your Python environment**:
   - Make sure TensorFlow is installed: `pip install tensorflow`
   - Install Pillow for image processing: `pip install pillow`

2. **Check image upload**:
   - Use smaller images (under 2MB)
   - Try standard image formats like JPEG or PNG
   - Make sure the image dimensions are reasonable (not too large)

3. **Check model compatibility**:
   - The app expects a model trained for multi-class classification of fresh/rotten items
   - Input shape should be (None, 160, 160, 3) - RGB images resized to 160x160
   - Output should match the number of classes in class_names.json

4. **Browser issues**:
   - Try a different browser
   - Clear browser cache
   - Check browser console for specific error messages

## Still Having Issues?

If you're still encountering problems, try the following:

1. Remove the existing models and use a fresh copy
2. Check all file permissions
3. Restart your computer to clear any lingering processes
4. Try running the model directly using `predict.py` script

```
python predict.py Test/freshtamto/test_image.jpg
```

This will help isolate whether the issue is with the model itself or the web interface.