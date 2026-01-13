import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random

# --- FIXED PARAMETER FOR OOM PREVENTION ---
# Reduced buffers for tf.data to minimize Pinned Host Memory usage
NUM_PARALLEL_CALLS = 4
PREFETCH_BUFFER_SIZE = 4


# --- 1. Model Components: Depthwise Separable Convolution Block ---

def separable_conv_block(inputs, filters, strides=1, name=None):
    """
    Creates a highly efficient Depthwise Separable Convolution block.
    This significantly reduces the number of parameters and computations compared to
    a standard Conv2D layer.
    """
    x = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        padding='same',
        strides=strides,
        use_bias=False,
        name=f'{name}_dw_conv'
    )(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    return x


# --- 2. The Efficient CNN Model (MobileNet-style) ---

def create_efficient_cnn(input_shape, num_classes):
    """Defines an efficient, MobileNet-style CNN architecture."""
    inputs = keras.Input(shape=input_shape)

    # 1. Initial Standard Convolution (to boost channel count quickly)
    x = layers.Conv2D(
        32, kernel_size=3, strides=2, padding='same', use_bias=False, name='initial_conv'
    )(inputs / 255.0)  # Normalize input here
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.ReLU(name='initial_relu')(x)  # Output: 64x64x32

    # 2. Sequential Depthwise Separable Blocks

    # Block 1: 64x64 -> 32x32
    x = separable_conv_block(x, 64, strides=2, name='block1_ds')
    x = separable_conv_block(x, 64, strides=1, name='block1_ds_repeat')

    # Block 2: 32x32 -> 16x16
    x = separable_conv_block(x, 128, strides=2, name='block2_ds')
    x = separable_conv_block(x, 128, strides=1, name='block2_ds_repeat')

    # Block 3: 16x16 -> 8x8
    x = separable_conv_block(x, 256, strides=2, name='block3_ds')
    x = separable_conv_block(x, 256, strides=1, name='block3_ds_repeat_a')
    x = separable_conv_block(x, 256, strides=1, name='block3_ds_repeat_b')

    # 3. Classifier Head

    # Global Average Pooling for final feature vector
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Dropout for regularization
    x = layers.Dropout(0.5, name='dropout')(x)

    # Dense layer for classification
    outputs = layers.Dense(num_classes, name='predictions')(x)

    return keras.Model(inputs, outputs, name='Efficient_CNN')


# --- 4. Data Loading and Filtering Functions (Kept Identical for stability) ---

def get_image_paths(data_dir, class_names):
    """Gathers all image file paths and their corresponding labels."""
    all_image_paths = []
    all_image_labels = []

    class_to_label = {name: i for i, name in enumerate(class_names)}
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path): continue

        label = class_to_label[class_name]

        for root, _, files in os.walk(class_path):
            for file_name in files:
                if file_name.lower().endswith(valid_extensions):
                    all_image_paths.append(os.path.join(root, file_name))
                    all_image_labels.append(label)

    return all_image_paths, all_image_labels


def decode_img_and_filter(img_path, img_size):
    """Reads, decodes, and resizes an image. Fails if corrupt."""
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)

    if tf.shape(img).shape[0] != 3:
        raise tf.errors.InvalidArgumentError(None, None, f"Invalid decoded shape for file: {img_path}")

    img = tf.image.resize(img, [img_size, img_size])
    return tf.cast(img, tf.float32)


def load_data(img_size, batch_size, data_dir="dataset", test_split_ratio=0.2):
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        return None, None, 0

    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        print("❌ ERROR: No class directories found inside 'dataset'.")
        return None, None, 0

    all_paths, all_labels = get_image_paths(data_dir, class_names)
    num_classes = len(class_names)

    if not all_paths:
        print("❌ ERROR: No images found in class directories.")
        return None, None, 0

    # 1. Split Paths
    dataset_size = len(all_paths)
    validation_size = int(test_split_ratio * dataset_size)

    indices = np.arange(dataset_size)
    np.random.seed(42)
    np.random.shuffle(indices)

    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    train_paths = all_paths[train_indices]
    train_labels = all_labels[train_indices]
    val_paths = all_paths[val_indices]
    val_labels = all_labels[val_indices]

    print(f"\nFound {dataset_size} files belonging to {num_classes} classes.")
    print(f"Using {len(train_paths)} files for training, {len(val_paths)} for validation.")
    print(f"Detected {num_classes} classes: {class_names}")

    # 2. Define the processing functions
    def process_path_with_filter(img_path, label):
        def py_filter_fn(path_tensor, label_tensor):
            path_str = path_tensor.numpy().decode('utf-8')
            try:
                img_tensor = decode_img_and_filter(path_str, img_size)
                # The image normalization is handled by the model itself now (inputs / 255.0)
                return True, img_tensor, label_tensor.numpy()
            except Exception as e:
                print(f"\n⚠️ Discarding corrupt file: {path_str}")
                return False, np.zeros((img_size, img_size, 3), dtype=np.float32), label_tensor.numpy()

        is_valid, img, label = tf.py_function(
            py_filter_fn, [img_path, label],
            [tf.bool, tf.float32, tf.int32]
        )

        is_valid.set_shape(())
        img.set_shape([img_size, img_size, 3])
        label.set_shape(())

        return is_valid, img, label

    # 3. Create TensorFlow Datasets
    train_ds_raw = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    test_ds_raw = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    # Apply Map, Filter, and Batch with OOM FIXES
    train_ds = (
        train_ds_raw
        .map(process_path_with_filter, num_parallel_calls=NUM_PARALLEL_CALLS)
        .filter(lambda is_valid, img, label: is_valid)
        .map(lambda is_valid, img, label: (img, label), num_parallel_calls=NUM_PARALLEL_CALLS)
        .batch(batch_size)
        .shuffle(buffer_size=1000)
        .prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
    )

    test_ds = (
        test_ds_raw
        .map(process_path_with_filter, num_parallel_calls=NUM_PARALLEL_CALLS)
        .filter(lambda is_valid, img, label: is_valid)
        .map(lambda is_valid, img, label: (img, label), num_parallel_calls=NUM_PARALLEL_CALLS)
        .batch(batch_size)
        .cache()
        .prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
    )

    setattr(train_ds, 'class_names', class_names)

    return train_ds, test_ds, num_classes


# --- 5. Experiment Runner (Main Logic) ---

def run_experiment(data_dir="dataset"):
    HPARAMS = {
        'img_size': 128,  # CRITICAL OOM FIX: Reduced input size
        'patch_size': 16,  # No longer used in CNN, kept for reference
        'embed_dim': 384,  # No longer used in CNN, kept for reference
        'batch_size': 4,  # Keep low for memory
        'learning_rate': 1e-4, 'epochs': 5
    }

    # 1. Load Data
    train_ds, test_ds, num_classes = load_data(HPARAMS['img_size'], HPARAMS['batch_size'], data_dir,
                                               test_split_ratio=0.2)
    if train_ds is None: return

    # 2. Data Augmentation
    data_augmentation = keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1)], name="data_augmentation")

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=PREFETCH_BUFFER_SIZE)

    # 3. Initialize Model and Build
    input_shape = (HPARAMS['img_size'], HPARAMS['img_size'], 3)
    cnn_model = create_efficient_cnn(input_shape, num_classes)

    print("\n✅ Efficient CNN Model Initialized (MobileNet-style)")
    print(f"Total trainable parameters: {cnn_model.count_params()}")

    # 4. Compile Model
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=HPARAMS['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 5. Model Checkpoint Callback
    # --- MODEL SAVE NAME CHANGE ---
    model_save_path = "best_efficient_cnn_model.keras"  # <--- NEW SAVE NAME
    checkpoint_callback = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks = [checkpoint_callback]

    # 6. Train Model
    print("\n--- Starting Efficient CNN Training ---")
    cnn_model.fit(
        train_ds,
        epochs=HPARAMS['epochs'],
        validation_data=test_ds,
        callbacks=callbacks
    )

    # 7. Evaluate Best Model
    print("\n--- Training Finished. Attempting to load and evaluate Best Saved Model ---")

    # Custom objects are not needed for this sequential CNN model
    best_model = None

    try:
        if os.path.exists(model_save_path):
            best_model = keras.models.load_model(model_save_path)
            print(f"✅ Successfully loaded best model from {model_save_path}.")
        else:
            print(f"⚠️ Best model file not found at {model_save_path}. Evaluating the final epoch's model instead.")
            best_model = cnn_model
    except Exception as e:
        print(f"❌ Failed to load best model. Error: {e}")
        print("Evaluating the final epoch's model instead.")
        best_model = cnn_model

    if best_model:
        loss, accuracy = best_model.evaluate(test_ds)
        print(f"Test Loss (Best Model): {loss:.4f}")
        print(f"Test Accuracy (Best Model): {accuracy * 100:.2f}%")


# --- Main Execution ---

if __name__ == "__main__":
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    run_experiment(data_dir="dataset")