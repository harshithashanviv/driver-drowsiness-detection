import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -----------------------------
# 1. Dataset Paths
# -----------------------------
train_path = "dataset/train"
test_path = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# 2. Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -----------------------------
# 3. Load MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

# -----------------------------
# 4. Add Custom Layers
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -----------------------------
# 5. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...")

# -----------------------------
# 6. Train Model
# -----------------------------
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

# -----------------------------
# 7. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/drowsiness_model.h5")

print("Model saved successfully!")