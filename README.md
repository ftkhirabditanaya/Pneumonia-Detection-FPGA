# Pneumonia Detection - FPGA Optimized

An efficient, FPGA-optimized pneumonia detection system using **MobileNetV2** and **TensorFlow Lite INT8 quantization**. This project enables fast medical imaging inference on edge devices and FPGA platforms.

## Overview

This project implements a deep learning model for pneumonia detection from chest X-ray images. The model is optimized for deployment on FPGA and edge devices using:
- **MobileNetV2** architecture for lightweight inference
- **INT8 quantization** for reduced model size and faster computation
- **TensorFlow Lite** for cross-platform deployment

## Features

- **Lightweight Model**: MobileNetV2-based architecture (efficient for edge devices)
- **INT8 Quantization**: Reduced model size and faster inference
- **TensorFlow Lite Support**: Easy deployment on mobile, edge, and FPGA devices
- **Binary Classification**: Normal vs. Pneumonia detection
- **Preprocessing Pipeline**: Automated image normalization and augmentation
- **Easy Inference**: Simple prediction script for testing

## Project Structure

```
PneumoniaDtection_FPGA/
├── main_fpga.py                     
├── convert_to_tflite.py             
├── predict_fpga.py                   
├── requirements.txt                  
├── pneumonia_mobilenetv2_fpga.h5     
├── pneumonia_mobilenetv2_int8.tflite 
├── data/
│   ├── Train/
│   │   ├── NORMAL/                   
│   │   └── PNEUMONIA/                
│   └── Test/
│       ├── NORMAL/                   
│       └── PNEUMONIA/                
└── README.md                        
```

## Requirements

- Python 3.8+ [ I'm using 3.11.0 as it' compatible with all of the python libraries]
- TensorFlow 2.20.0
- NumPy 1.26.4
- Pillow 12.1.0

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ftkhirabditanaya/Pneumonia-Detection-FPGA.git
   cd Pneumonia-Detection-FPGA
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Model Architecture

### MobileNetV2 + Custom Classifier

```
Input (224×224×3)
    ↓
MobileNetV2 (pretrained on ImageNet - frozen)
    ↓
Global Average Pooling
    ↓
Dense (256 units) + ReLU + Dropout(0.5)
    ↓
Dense (1 unit) + Sigmoid
    ↓
Output (Binary Classification)
```

**Key Features**:
- Pretrained ImageNet weights for better generalization
- Transfer learning to reduce training time
- Dropout regularization to prevent overfitting
- Binary output using sigmoid activation

## Usage

### 1. Train the Model

```bash
python main_fpga.py
```

This script will:
- Load training and test data from `data/Train` and `data/Test`
- Resize images to 224×224 pixels
- Train the MobileNetV2 model with binary classification
- Save the trained model as `pneumonia_mobilenetv2_fpga.h5`

### 2. Convert to TFLite INT8

```bash
python convert_to_tflite.py
```

This script converts the trained Keras model to TensorFlow Lite format with INT8 quantization:
- Reduces model size significantly
- Improves inference speed on edge devices
- Generates `pneumonia_mobilenetv2_int8.tflite`

### 3. Run Inference

```bash
python predict_fpga.py
```

This script:
- Loads the quantized TFLite model
- Preprocesses X-ray images
- Performs INT8 quantization on input
- Returns prediction (0 = Normal, 1 = Pneumonia)

**Example Output**:
```
Input shape expected: [1, 224, 224, 3]
Input dtype expected: int8
Output shape: [1, 1]
Predicted output: 0.95 (Pneumonia detected with 95% confidence)
```

## Dataset

The project uses chest X-ray images organized as follows:
- **Training**: `data/Train/NORMAL/` and `data/Train/PNEUMONIA/`
- **Testing**: `data/Test/NORMAL/` and `data/Test/PNEUMONIA/`

Images are automatically:
- Resized to 224×224 pixels
- Normalized to [0, 255] range
- Converted from grayscale to RGB (3-channel replication)

## Model Performance

| Metric | Value |
|--------|-------|
| Input Size | 224×224×3 |
| Model Size (Keras) | ~9 MB |
| Model Size (TFLite INT8) | ~2-3 MB |
| Inference Time | <100ms (on CPU) |
| Output | Binary (Normal/Pneumonia) |

## File Descriptions

| File | Purpose |
|------|---------|
| `main_fpga.py` | Trains MobileNetV2 model on pneumonia dataset |
| `convert_to_tflite.py` | Quantizes model to TFLite INT8 format |
| `predict_fpga.py` | Runs inference on X-ray images |
| `pneumonia_mobilenetv2_fpga.h5` | Trained Keras model (high precision) |
| `pneumonia_mobilenetv2_int8.tflite` | Quantized TFLite model (edge deployment) |

## INT8 Quantization Benefits

1.**60-75% smaller model size**

2.**2-3x faster inference** on CPU

3.**Minimal accuracy loss** (~1-2%)

4.**FPGA/Edge device compatible**

5.**Lower memory footprint**

## FPGA Deployment

To deploy on FPGA:
1. Use the `.tflite` model file
2. Implement quantization-aware inference
3. Map operations to FPGA hardware units
4. Consider frameworks like:
   - ONNX Runtime
   - TensorFlow Lite Micro
   - Custom FPGA implementations

## Image Preprocessing Pipeline

```python
# Load X-ray image
img = Image.open(path).convert("L")  # Grayscale

# Resize
img = img.resize((224, 224))

# Convert to RGB (replicate channels)
img = img.convert("RGB")

# INT8 Quantization
scale, zero_point = input_details[0]['quantization']
img = img / scale + zero_point
```

## Troubleshooting

### Model not found error
- Ensure trained model files exist in the project directory
- Run `main_fpga.py` first to generate the models

### Out of memory errors
- Reduce batch size in `main_fpga.py`
- Use smaller input images (224×224 is already optimized)

### Prediction accuracy issues
- Verify image preprocessing matches training pipeline
- Check image format (grayscale → RGB conversion)
- Ensure proper INT8 quantization parameters

## License

This project is open source. Feel free to use, modify, and distribute.

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Lite Quantization](https://www.tensorflow.org/lite/performance/quantization)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)

## Acknowledgments

- TensorFlow and Keras teams for excellent deep learning frameworks
- Medical imaging community for dataset contributions
- FPGA optimization insights from edge computing research

---

