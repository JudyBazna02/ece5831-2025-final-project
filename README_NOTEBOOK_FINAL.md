# ASL Sign Recognition System - Project Documentation

## Overview

This Jupyter notebook demonstrates a complete Real-Time American Sign Language (ASL) recognition system. The notebook includes model training, live recognition testing, and confusion matrix generation for performance analysis.

---

## Contents

The notebook is organized into the following sections:

1. **Library Imports** - TensorFlow and required dependencies
2. **Model Training** - CNN training on ASL image dataset
3. **Model Testing** - Screenshots and testing results
4. **Confusion Matrix** - Performance visualization from test data

---

## Section 1: Library Imports

### TensorFlow Import
```python
import tensorflow as tf
```

**Purpose:** Imports TensorFlow framework for deep learning model development and training.

---

## Section 2: Model Training

### Dataset Information
- **Training samples:** 4,160 images
- **Validation samples:** 1,040 images
- **Total classes:** 26 (letters A-Z)
- **Validation split:** 20% (0.2)

### Data Augmentation
The training pipeline applies the following augmentation techniques:
- **Rescaling:** Normalizes pixel values to 0-1 range
- **Rotation:** Random rotation up to 20 degrees
- **Width shift:** Random horizontal shift of 20%
- **Height shift:** Random vertical shift of 20%
- **Horizontal flip:** Random horizontal flipping

### CNN Architecture

**Layer Structure:**

1. **First Convolutional Block**
   - Conv2D: 32 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size
   - Output shape: (None, 31, 31, 32)

2. **Second Convolutional Block**
   - Conv2D: 64 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size
   - Output shape: (None, 14, 14, 64)

3. **Third Convolutional Block**
   - Conv2D: 128 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D: 2x2 pool size
   - Output shape: (None, 6, 6, 128)

4. **Dense Layers**
   - Flatten: Converts 2D features to 1D
   - Dense: 128 units, ReLU activation
   - Dropout: 30% dropout rate
   - Output Dense: 26 units, softmax activation

**Total Parameters:** 686,554 trainable parameters

### Training Configuration

- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Categorical crossentropy
- **Metrics:** Accuracy
- **Epochs:** 30
- **Batch Size:** 32
- **Input Size:** 64x64x3 (RGB images)

### Training Results

**Model Performance:**
- Training completed successfully over 30 epochs
- Progress tracked through loss and accuracy metrics
- Model saved after training completion

---

## Section 3: Model Testing

### Testing Screenshots

The notebook includes visual documentation of the system in operation:
- Real-time webcam recognition interface
- Live prediction display with confidence scores
- Sentence building functionality demonstration

**Features Demonstrated:**
- Letter prediction with confidence values
- Real-time frame processing
- Bounding box visualization
- Sentence construction from detected signs

---

## Section 4: Confusion Matrix Generation

### Data Source
- **Input file:** CSV file containing test results
- **Format:** CSV with true_label and pred_label columns

### Matrix Generation Process

**Required Libraries:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import string
```

**Steps:**
1. Load CSV data containing true and predicted labels
2. Define class labels (A-Z) using string.ascii_uppercase
3. Convert labels to string type for consistency
4. Compute confusion matrix using sklearn
5. Convert matrix to pandas DataFrame
6. Visualize using seaborn heatmap

### Visualization Configuration

**Plot Settings:**
- Figure size: 14x12 inches
- Color map: Blues
- Annotations: Disabled for large dataset
- Format: Integer values
- Axes labeled with predicted and true letters
- Title: "Confusion Matrix for ASL Letter Prediction"

**Interpretation:**
- Rows represent true ASL letters
- Columns represent predicted ASL letters
- Diagonal values show correct predictions
- Off-diagonal values indicate misclassifications
- Darker colors indicate higher prediction counts

---

## Requirements

### Python Version
- Python 3.8 or higher recommended

### Required Libraries
```
tensorflow>=2.8.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
```

### Installation
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python
```

---

## File Structure

```
project/
├── final-project.ipynb        # This notebook
├── asl_model.h5               # Saved trained model
├── confdata.csv               # Test results for confusion matrix
└── ASL_images/                # Dataset directory
    ├── A/
    ├── B/
    └── ... (26 letter folders)
```

---

## Usage Instructions

### Running the Notebook

**Step 1: Open Jupyter Notebook**
```bash
jupyter notebook final-project.ipynb
```

**Step 2: Execute Cells Sequentially**
- Run Cell 1: Import TensorFlow
- Run Cell 2: Train model (approximately 30-40 minutes)
- Run Cell 3: View testing screenshots
- Run Cell 4: Generate confusion matrix

### Model Training
- Ensure dataset path is correctly set to your local directory
- Training will take 30-40 minutes on CPU
- Model automatically saved after training completes
- Monitor loss and accuracy during training

### Confusion Matrix
- Requires CSV file with test results
- CSV must contain columns: true_label, pred_label
- Matrix visualizes model performance across all classes
- Identifies commonly confused letter pairs

---

## Dataset Requirements

### Structure
Images must be organized in labeled folders:
```
ASL_images/
├── A/ (200 images)
├── B/ (200 images)
├── C/ (200 images)
...
└── Z/ (200 images)
```

### Image Specifications
- **Format:** JPG, JPEG, or PNG
- **Resolution:** Any (resized to 64x64 during preprocessing)
- **Background:** Plain/solid color recommended
- **Lighting:** Consistent, well-lit conditions
- **Content:** Clear hand signs, centered in frame

---

## Model Performance

### Expected Accuracy
- **Training Accuracy:** 90-95%
- **Validation Accuracy:** 85-92%
- **Test Accuracy:** Varies based on testing conditions

### Performance Factors
**High Accuracy Letters:**
- Letters with distinct hand shapes (C, D, I, O, Y)
- Signs with unique finger positions

**Challenging Letters:**
- Similar fist shapes (A, E, M, N, S, T)
- Small variations in thumb position
- Letters requiring motion (J, Z) when captured as static images

---

## Troubleshooting

### Common Issues

**Issue: "Data directory not found"**
- Solution: Update dataset path in training cell to match your directory structure

**Issue: Low training accuracy**
- Solution: Increase number of epochs (30-50)
- Solution: Verify dataset quality and balance
- Solution: Check for mislabeled images

**Issue: Confusion matrix shows poor performance**
- Solution: Retrain with more epochs
- Solution: Improve dataset quality
- Solution: Add more training images per class
- Solution: Enhance data augmentation parameters

**Issue: Out of memory during training**
- Solution: Reduce batch size from 32 to 16
- Solution: Reduce image size from 64 to 48
- Solution: Close other applications

---

## Technical Notes

### Data Augmentation Strategy
Applied augmentation techniques help the model generalize by:
- Accounting for hand rotation variations
- Handling different hand positions in frame
- Simulating both left and right hand signing
- Preventing overfitting to specific orientations

### Model Architecture Rationale
- **Progressive filters (32→64→128):** Captures increasingly complex features
- **Max pooling:** Reduces spatial dimensions while retaining key information
- **ReLU activation:** Introduces non-linearity for pattern learning
- **Dropout:** Prevents overfitting by randomly disabling neurons
- **Softmax output:** Produces probability distribution over classes

### Training Process
1. Images loaded from directory structure
2. Augmentation applied during training
3. Model learns feature extraction and classification
4. Validation monitors generalization
5. Best weights saved for inference

---

## Results Analysis

### Confusion Matrix Insights
The confusion matrix reveals:
- Which letters are most accurately recognized
- Common misclassification patterns
- Letters that require additional training data
- Model confidence across different sign types

### Performance Optimization
Based on confusion matrix results:
1. Identify poorly performing letter pairs
2. Add more training samples for confused classes
3. Adjust augmentation parameters
4. Consider ensemble methods for similar signs

---

### pre-recorded presentation video: https://drive.google.com/file/d/110NdGaswzM0AgIVjg0pvyl05tsp_qwE7/view?usp=sharing
### presentation slides:  https://docs.google.com/presentation/d/1gXLJ5y5n401QhLM3PpLJnmN34NtI8Ue0/edit?usp=sharing&ouid=100932542296015576315&rtpof=true&sd=true
### report: https://docs.google.com/document/d/1uI-3x9Atoq-wqVQyTz1-86_PjlzWjxQ2/edit?usp=sharing&ouid=100932542296015576315&rtpof=true&sd=true
### The dataset: https://drive.google.com/file/d/1PPsVCpfB_q-JT7LR3s1_M5jBfHeID35n/view?usp=sharing
### The demo video: https://youtu.be/9ySVdMwImQo?si=CZT0nCGpxdS_LqoB
