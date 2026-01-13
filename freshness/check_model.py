import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import argparse

def check_environment():
    """Check the TensorFlow environment"""
    print("\n=== Environment Information ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Python version: {sys.version}")
    print(f"CUDA available: {tf.config.list_physical_devices('GPU')}")
    print(f"Current directory: {os.getcwd()}")

def find_models():
    """Find all model files in the current directory"""
    print("\n=== Model Files ===")
    model_extensions = ['.h5', '.keras', '.pb']
    models_found = []
    
    # Look in current directory and common subdirectories
    search_paths = ['.', 'models', 'final_saved_model']
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
        
        if os.path.isdir(path):
            # Check if this is a SavedModel directory
            if os.path.exists(os.path.join(path, 'saved_model.pb')):
                models_found.append((path, 'saved_model'))
            
            # Check for files with model extensions
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path) and any(file.endswith(ext) for ext in model_extensions):
                    models_found.append((file_path, 'file'))
        elif any(os.path.basename(path).endswith(ext) for ext in model_extensions):
            models_found.append((path, 'file'))
    
    if not models_found:
        print("❌ No model files found")
    else:
        print(f"Found {len(models_found)} potential model file(s):")
        for model_path, model_type in models_found:
            print(f" - {model_path} ({model_type})")
    
    return models_found

def check_class_names():
    """Find and validate class names file"""
    print("\n=== Class Names File ===")
    class_names_path = 'class_names.json'
    
    if not os.path.exists(class_names_path):
        print(f"❌ Class names file not found at {class_names_path}")
        return None
    
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        if not isinstance(class_names, list):
            print(f"❌ Class names file does not contain a list: {type(class_names)}")
            return None
        
        print(f"✓ Found class names file with {len(class_names)} classes")
        print(f"Classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        return class_names
    except Exception as e:
        print(f"❌ Error reading class names file: {str(e)}")
        return None

def try_load_model(model_path, model_type='file'):
    """Try to load a model from the given path"""
    print(f"\n=== Testing Model Loading: {model_path} ===")
    try:
        if model_type == 'file':
            model = tf.keras.models.load_model(model_path)
            print(f"✓ Successfully loaded model from {model_path}")
            
            # Show model summary
            print("\nModel summary:")
            model.summary()
            
            # Show input and output shapes
            input_shape = model.input_shape
            output_shape = model.output_shape
            print(f"\nInput shape: {input_shape}")
            print(f"Output shape: {output_shape}")
            
            return model, input_shape, output_shape
        else:
            # Try to load as SavedModel
            model = tf.saved_model.load(model_path)
            print(f"✓ Successfully loaded SavedModel from {model_path}")
            return model, None, None
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def test_prediction(model, class_names=None):
    """Test the model with a dummy input"""
    print("\n=== Testing Model Prediction ===")
    try:
        # Create a dummy input (zeros)
        input_shape = model.input_shape
        if not input_shape:
            print("❌ Unable to determine input shape")
            return False
        
        # Strip away the batch dimension if it's None
        if input_shape[0] is None:
            input_shape = input_shape[1:]
        
        # Create tensor of appropriate shape
        dummy_input = np.zeros((1,) + input_shape)
        print(f"Created dummy input with shape: {dummy_input.shape}")
        
        # Run prediction
        print("Running prediction with zeros array...")
        predictions = model.predict(dummy_input, verbose=0)
        
        print(f"✓ Model successfully made predictions. Output shape: {predictions.shape}")
        
        if class_names:
            # Get top prediction
            top_class_idx = np.argmax(predictions[0])
            if top_class_idx < len(class_names):
                print(f"Top prediction for zeros input: {class_names[top_class_idx]} ({predictions[0][top_class_idx]:.4f})")
            else:
                print(f"⚠️ Warning: Predicted class index {top_class_idx} is outside the range of class names")
        
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Check and diagnose model issues")
    parser.add_argument("--model", help="Path to specific model to check")
    args = parser.parse_args()
    
    print("===== Freshness Model Diagnostic Tool =====")
    
    # Check environment
    check_environment()
    
    # Find models or use specified model
    if args.model:
        models_found = [(args.model, 'file')]
        print(f"\nUsing specified model: {args.model}")
    else:
        models_found = find_models()
    
    # Check class names
    class_names = check_class_names()
    
    # Try to load each model
    for model_path, model_type in models_found:
        model, input_shape, output_shape = try_load_model(model_path, model_type)
        if model:
            # Try a test prediction
            test_prediction(model, class_names)
    
    print("\n===== Diagnostic Complete =====")
    print("If you found issues, consider:")
    print("1. Ensuring the model and class_names.json files are in the same directory as the script")
    print("2. Checking the model format is compatible with your TensorFlow version")
    print("3. Verifying the model's input shape matches your preprocessing")
    print("4. If using GPU, checking that CUDA and cuDNN are properly installed")

if __name__ == "__main__":
    main()