import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random

# --- FIXED PARAMETER FOR OOM PREVENTION ---
# Use a small, fixed number for parallel calls and prefetch buffer size.
NUM_PARALLEL_CALLS = 4
PREFETCH_BUFFER_SIZE = 4


# --- 1. Model Components: Patch Embedding ---

class PatchEmbedding(layers.Layer):
    """Splits image into patches, projects them, and adds positional embeddings."""

    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

        self.position_embedding = None
        self.num_patches = None

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.num_patches = (height // self.patch_size) * (width // self.patch_size)

        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.embed_dim
        )
        super().build(input_shape)

    def call(self, images):
        x = images / 255.0
        x = self.projection(x)
        x = self.flatten(x)

        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embed = self.position_embedding(positions)

        x = x + pos_embed[tf.newaxis, ...]
        return x


# --- 2. Model Components: Shared Recursive Block (R) Helper ---

def create_trm_layer(embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
    """Instantiates Keras layers for the single shared Recursive Refinement Block (R)."""
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
    )
    dense_projection_1 = layers.Dense(mlp_dim, activation="gelu")
    dense_projection_2 = layers.Dense(embed_dim)
    norm_attn = layers.LayerNormalization(epsilon=1e-6)
    norm_mlp = layers.LayerNormalization(epsilon=1e-6)
    dropout = layers.Dropout(dropout_rate)

    def trm_block_func(x, attn_layer, dense1, dense2, norm1, norm2, drop_layer):
        # Attention Block (Pre-norm residual)
        norm_output_1 = norm1(x)
        attn = attn_layer(norm_output_1, norm_output_1)
        x = x + drop_layer(attn)
        # MLP Block (Pre-norm residual)
        norm_output_2 = norm2(x)
        mlp_output = dense2(drop_layer(dense1(norm_output_2)))
        x = x + drop_layer(mlp_output)
        return x

    return (trm_block_func, attn_output, dense_projection_1, dense_projection_2,
            norm_attn, norm_mlp, dropout)


# --- 3. The TRM-Vision Model (TRM_Vision) ---

class TRMVision(keras.Model):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, depth, num_heads, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)

        (self.trm_block_func, self.attn, self.dense1, self.dense2, self.norm_attn,
         self.norm_mlp, self.dropout_layer) = create_trm_layer(embed_dim, num_heads, mlp_dim)

        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        self.classifier = layers.Dense(num_classes)

    def call(self, inputs):
        x = self.patch_embed(inputs)
        S_t = x
        for _ in range(self.depth):
            S_t = self.trm_block_func(
                S_t, self.attn, self.dense1, self.dense2, self.norm_attn, self.norm_mlp, self.dropout_layer
            )

        S_T = self.final_norm(S_t)
        pooled_output = tf.reduce_mean(S_T, axis=1)  # Global Average Pooling
        logits = self.classifier(pooled_output)

        return logits


# --- 4. Data Loading and Filtering Functions (Corruption/OOM Fixes) ---

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

        # Use os.walk for recursive search
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

    # Check shape to ensure it's a valid 3D image (H, W, C)
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

    # 1. Split Paths into Training and Validation Sets
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
        # Function wrapper for filtering
        def py_filter_fn(path_tensor, label_tensor):
            path_str = path_tensor.numpy().decode('utf-8')
            try:
                # Attempt to decode and process the image (Corruption Check)
                img_tensor = decode_img_and_filter(path_str, img_size)
                return True, img_tensor, label_tensor.numpy()
            except Exception as e:
                print(f"\n⚠️ Discarding corrupt file: {path_str}")
                # Return placeholder if corrupt
                return False, np.zeros((img_size, img_size, 3), dtype=np.float32), label_tensor.numpy()

        is_valid, img, label = tf.py_function(
            py_filter_fn, [img_path, label],
            [tf.bool, tf.float32, tf.int32]
        )

        is_valid.set_shape(())
        img.set_shape([img_size, img_size, 3])
        label.set_shape(())

        return is_valid, img, label

    # 3. Create TensorFlow Datasets from File Paths and Labels
    train_ds_raw = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    test_ds_raw = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    # --- Apply Map, Filter, and Batch with OOM FIX ---

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
        'img_size': 128,  # <-- CRITICAL OOM FIX: Reduced from 224 to 128
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 6, 'num_heads': 6, 'mlp_dim': 1536,
        'batch_size': 4,
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
    trm_model = TRMVision(
        img_size=HPARAMS['img_size'], patch_size=HPARAMS['patch_size'], num_classes=num_classes,
        embed_dim=HPARAMS['embed_dim'], depth=HPARAMS['depth'], num_heads=HPARAMS['num_heads'],
        mlp_dim=HPARAMS['mlp_dim']
    )
    trm_model.build(input_shape=(HPARAMS['batch_size'], HPARAMS['img_size'], HPARAMS['img_size'], 3))

    print("\n✅ TRM-Vision Model Initialized")
    print(f"Total trainable parameters: {trm_model.count_params()}")

    # 4. Compile Model
    trm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=HPARAMS['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 5. Model Checkpoint Callback (Saves Best Model)
    model_save_path = "best_trm_fresh_rotten_model.keras"
    # The 'verbose=1' argument ensures Keras prints a message every time the model is saved.
    checkpoint_callback = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1  # This prints the save action message to the console.
    )
    callbacks = [checkpoint_callback]

    # 6. Train Model
    print("\n--- Starting End-to-End Training (Final OOM Fixes Active) ---")
    trm_model.fit(
        train_ds,
        epochs=HPARAMS['epochs'],
        validation_data=test_ds,
        callbacks=callbacks
    )

    # 7. Evaluate Best Model
    print("\n--- Training Finished. Attempting to load and evaluate Best Saved Model ---")

    custom_objects = {"PatchEmbedding": PatchEmbedding, "TRMVision": TRMVision}
    best_model = None

    try:
        # Check if the file exists before attempting to load
        if os.path.exists(model_save_path):
            best_model = keras.models.load_model(model_save_path, custom_objects=custom_objects)
            print(f"✅ Successfully loaded best model from {model_save_path}.")
        else:
            print(f"⚠️ Best model file not found at {model_save_path}. Evaluating the final epoch's model instead.")
            best_model = trm_model
    except Exception as e:
        print(f"❌ Failed to load best model from {model_save_path}. Error: {e}")
        print("Evaluating the final epoch's model instead.")
        best_model = trm_model

    if best_model:
        loss, accuracy = best_model.evaluate(test_ds)
        print(f"Test Loss (Best Model): {loss:.4f}")
        print(f"Test Accuracy (Best Model): {accuracy * 100:.2f}%")


# --- Main Execution ---

if __name__ == "__main__":
    # Setting inter- and intra-op parallelism threads might help manage CPU usage related to I/O
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    run_experiment(data_dir="dataset")