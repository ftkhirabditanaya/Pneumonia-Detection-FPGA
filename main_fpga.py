# ===============================
# Pneumonia Detection (FPGA Optimized MobileNetV2)
# ===============================

# ---- 1. IMPORT LIBRARIES ----
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

print("TensorFlow Version:", tf.__version__)

# ---- 2. IMAGE DATA GENERATORS (MINIMAL) ----
train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ---- 3. LOAD DATA ----
train_generator = train_datagen.flow_from_directory(
    'data/Train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'data/Test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# ---- 4. LOAD PRETRAINED MOBILENETV2 ----
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze pretrained layers
base_model.trainable = False

# ---- 5. BUILD LIGHTWEIGHT CLASSIFIER ----
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # Minimal head for FPGA
])

# ---- 6. COMPILE MODEL ----
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---- 7. TRAIN MODEL ----
print("\n>>> Training Started <<<\n")

model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# ---- 8. SAVE MODEL ----
model.save("pneumonia_mobilenetv2_fpga.h5")
print("\n✅ Model saved for FPGA optimization")
