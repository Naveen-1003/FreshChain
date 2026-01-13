import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def predict_image(model, image_path, class_names):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    score = predictions[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(score)[-3:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_scores = [score[i] for i in top_indices]
    
    # Display the image and predictions
    plt.figure(figsize=(8, 6))
    plt.imshow(img_array)
    plt.axis('off')
    plt.title(f"Prediction: {top_classes[0]} ({top_scores[0]*100:.2f}%)")
    
    # Add a text box with the top 3 predictions
    pred_text = "\n".join([f"{cls}: {scr*100:.2f}%" for cls, scr in zip(top_classes, top_scores)])
    plt.figtext(0.15, 0.05, pred_text, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    
    # Save and show the result
    result_path = 'prediction_result.png'
    plt.savefig(result_path)
    print(f"Result saved to {result_path}")
    
    return {
        'top_class': top_classes[0],
        'top_score': float(top_scores[0]),
        'all_predictions': [(cls, float(scr)) for cls, scr in zip(top_classes, top_scores)]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict freshness of fruits/vegetables")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model", default="final_model.h5", help="Path to the model file")
    args = parser.parse_args()
    
    # Load model
    model = keras.models.load_model(args.model)
    
    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Make prediction
    result = predict_image(model, args.image_path, class_names)
    print(f"Top prediction: {result['top_class']} with {result['top_score']*100:.2f}% confidence")