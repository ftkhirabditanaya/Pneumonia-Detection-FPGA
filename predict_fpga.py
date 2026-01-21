# ==========================================================
# Pneumonia Detection - FPGA / TFLite INT8 Inference
# Single-output Sigmoid Model (CORRECTED)
# ==========================================================

import numpy as np
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------
# 1. LOAD TFLITE MODEL
# ----------------------------------------------------------
interpreter = tf.lite.Interpreter(
    model_path="pneumonia_mobilenetv2_int8.tflite"
)
interpreter.allocate_tensors()

# ----------------------------------------------------------
# 2. GET INPUT / OUTPUT DETAILS
# ----------------------------------------------------------
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape expected :", input_details[0]['shape'])
print("Input dtype expected :", input_details[0]['dtype'])
print("Output shape        :", output_details[0]['shape'])

# ----------------------------------------------------------
# 3. LOAD & PREPROCESS X-RAY IMAGE
# ----------------------------------------------------------
img_path = "data/Test/NORMAL/IM-0001-0001.jpeg"

# Load grayscale X-ray
img = Image.open(img_path).convert("L")

# Resize
img = img.resize((224, 224))

# Convert grayscale → RGB (3-channel replication)
img = img.convert("RGB")

# Convert to numpy
img = np.array(img)

# ----------------------------------------------------------
# 4. INT8 INPUT QUANTIZATION
# ----------------------------------------------------------
scale, zero_point = input_details[0]['quantization']

img = img / scale + zero_point
img = np.clip(img, 0, 255).astype(np.uint8)

# Add batch dimension
img = np.expand_dims(img, axis=0)

# ----------------------------------------------------------
# 5. SET INPUT & RUN INFERENCE
# ----------------------------------------------------------
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# ----------------------------------------------------------
# 6. GET OUTPUT & DEQUANTIZE
# ----------------------------------------------------------
output = interpreter.get_tensor(output_details[0]['index'])

out_scale, out_zero_point = output_details[0]['quantization']
probability = (output[0][0] - out_zero_point) * out_scale

# ----------------------------------------------------------
# 7. DECISION LOGIC (SIGMOID OUTPUT)
# ----------------------------------------------------------
print(f"Pneumonia probability: {probability:.4f}")

if probability >= 0.5:
    print("🫁 Pneumonia Detected")
else:
    print("✅ Normal")
