# ===============================
# Script to convert Keras model to TensorFlow Lite format with quantization for FPGA deployment
# ===============================

import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("pneumonia_mobilenetv2_fpga.h5")

# Representative dataset (VERY IMPORTANT)
def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open("pneumonia_mobilenetv2_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model created")


