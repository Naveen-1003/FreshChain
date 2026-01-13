import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
TEST_DIR = 'Test'
MODEL_PATH = 'best_model.h5'
CLASS_NAMES_PATH = 'class_names.json'
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"TensorFlow version: {tf.__version__}")
print(f"Using model path: {MODEL_PATH}")


# --- Utility Functions ---

def load_class_names(filepath):
    """Loads all 18 class names from the saved JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Class names file not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} total class names (from training).")
    return class_names


def load_test_dataset(directory, img_size, batch_size, seed):
    """
    Loads the test dataset. The raw_ds will only contain 14 classes' metadata.
    """
    if not os.path.exists(directory):
        print(f"Error: Test directory not found at {directory}")
        return None, None

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    prefetched_ds = raw_ds.prefetch(tf.data.AUTOTUNE)

    return raw_ds, prefetched_ds


def get_labels_and_predictions(model, raw_ds, prefetched_ds, all_class_names):
    """
    Generates true labels and adjusted model predictions for only the 14 classes found.
    """
    y_true = []

    # 1. Get true labels (as 14-element one-hot vectors or indices)
    for _, labels in prefetched_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    # 2. Get predictions (as 18-element probability vectors)
    print("Generating predictions...")
    y_pred_probs_18 = model.predict(prefetched_ds, verbose=1)

    # 3. Handle Mismatching Class Names (Typo Correction)
    test_class_names_raw = raw_ds.class_names
    test_class_names_corrected = []

    # Map the folder names to the correct names used in the 18-class training list
    for name in test_class_names_raw:
        if name == 'freshpatato':
            corrected_name = 'freshpotato'
        elif name == 'rottenpatato':
            corrected_name = 'rottenpotato'
        elif name == 'freshtamto':
            corrected_name = 'freshtomato'
        elif name == 'rottentamto':
            corrected_name = 'rottentomato'
        else:
            corrected_name = name
        test_class_names_corrected.append(corrected_name)

    # 4. Determine which of the 18 indices correspond to the 14 test classes (using corrected names)
    try:
        test_indices_in_model = [all_class_names.index(name) for name in test_class_names_corrected]
    except ValueError as e:
        print(f"\nFATAL ERROR: A class name mismatch still exists. Error detail: {e}")
        print("Please check the names in your 'class_names.json' against your test folder names.")
        raise

    print(f"Mapped 14 test classes to 18 model outputs using indices: {test_indices_in_model}")

    # 5. Filter the 18 predictions down to the 14 relevant columns
    # This creates a (N, 14) array of probabilities
    y_pred_probs_14 = y_pred_probs_18[:, test_indices_in_model]

    # 6. Get the final predicted class index (0-13) based on the filtered probabilities
    y_pred = np.argmax(y_pred_probs_14, axis=1)

    # Return the corrected names for the final report
    return np.array(y_true), y_pred, test_class_names_corrected


def plot_confusion_matrix(cm, class_names):
    """Plots the confusion matrix using Seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.title('Confusion Matrix (Only 14 Classes Evaluated)', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---

def run_test_script():
    # 1. Load Model and Class Names (All 18)
    if not os.path.exists(MODEL_PATH):
        print(f"\nFATAL ERROR: Model file not found at {MODEL_PATH}")
        return

    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load the model from {MODEL_PATH}. Details: {e}")
        return

    all_class_names = load_class_names(CLASS_NAMES_PATH)
    if all_class_names is None:
        return

    # 2. Load Test Data (Only 14 folders found)
    print("\n--- Loading Test Data ---")
    raw_test_ds, test_ds = load_test_dataset(TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED)
    if raw_test_ds is None:
        return

    num_dataset_classes = len(raw_test_ds.class_names)
    print(f"Found {num_dataset_classes} folders in the test directory.")

    # 3. Detailed Predictions and Reporting
    print("\n--- Detailed Prediction and Reporting (14 Classes) ---")

    # The key function call that handles mapping and prediction
    try:
        y_true, y_pred, test_class_names_corrected = get_labels_and_predictions(
            model, raw_test_ds, test_ds, all_class_names
        )
    except Exception as e:
        # The specific ValueError is now handled inside the function,
        # but catching a general error here is good practice.
        return

        # 4. Classification Report
    print("\n--- Classification Report ---")

    # We use the corrected names for the final report
    print(classification_report(y_true, y_pred, target_names=test_class_names_corrected, zero_division=0))

    # 5. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)

    # Print and plot the matrix for the 14 classes
    print(cm)
    plot_confusion_matrix(cm, test_class_names_corrected)


if __name__ == '__main__':
    run_test_script()