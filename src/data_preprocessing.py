import os
import tensorflow as tf

# ==============================
# Dataset Path
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "..", "dataset")

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# ==============================
# Image Parameters
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading and preprocessing dataset...\n")

# ==============================
# Load Training Dataset
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# ==============================
# Load Validation Dataset
# ==============================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# ==============================
# Load Test Dataset
# ==============================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# ==============================
# Normalize (Rescale) Images
# ==============================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# ==============================
# Display Information
# ==============================

class_names = train_dataset.class_names

print("\nPreprocessing Complete ✅")
print("Classes:", class_names)
print("Training batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))
print("Test batches:", len(test_dataset))