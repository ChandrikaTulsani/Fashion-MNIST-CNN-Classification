# Fashion-MNIST CNN Classification

A deep learning project that classifies grayscale clothing images into 10 categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. Trained on the Fashion-MNIST dataset over 20 epochs, achieving 91.40% test accuracy.

---

## Overview

Fashion-MNIST is a benchmark dataset for image classification, consisting of 70,000 grayscale images of clothing items across 10 categories. This project builds a CNN from scratch, trains it with dropout regularization, and analyzes training behavior — including the identification of overfitting from epoch 5 onward.

---

## Dataset

**Fashion-MNIST** — loaded directly via `tensorflow.keras.datasets`.

| Property | Value |
|---|---|
| Training Samples | 60,000 |
| Test Samples | 10,000 |
| Image Size | 28 × 28 pixels (grayscale) |
| Classes | 10 |
| Input Shape | (28, 28, 1) |

**Classes:**

| Label | Category |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Project Structure

```
├── Lab_assessment1.ipynb    # Main notebook
└── README.md
```

> Dataset is downloaded automatically via Keras — no manual download required.

---

## Methodology

### 1. Data Loading & Preprocessing
- Dataset loaded using `tensorflow.keras.datasets.fashion_mnist`
- Pixel values normalized from [0, 255] to [0, 1]
- Images reshaped from (28, 28) to (28, 28, 1) to match CNN input requirements
- Seeds set for reproducibility (`numpy` and `tensorflow`, seed=42)

### 2. Model Architecture

```
Conv2D(32, 3×3, relu)
    ↓
Conv2D(64, 3×3, relu)
    ↓
MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(128, relu)
    ↓
Dropout(0.3)
    ↓
Dense(10, softmax)
```

### 3. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 20 |
| Batch Size | 64 |
| Validation Split | 20% (of training data) |

### 4. Evaluation
- Training vs validation accuracy and loss plotted across all 20 epochs
- Overfitting analysis — divergence between train and val loss identified from epoch 5
- Final evaluation on the held-out test set

---

## Results

### Training Progress (Selected Epochs)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 83.53% | 88.57% | 0.4639 | 0.3108 |
| 5 | 93.72% | 91.51% | 0.1707 | 0.2470 |
| 10 | 97.20% | 91.78% | 0.0751 | 0.3175 |
| 15 | 98.24% | 91.84% | 0.0487 | 0.3821 |
| 20 | 98.77% | 92.18% | 0.0338 | 0.4358 |

### Final Test Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **91.40%** |

### Overfitting Observation
From epoch 5 onward, training accuracy continues to climb while validation loss begins increasing — a classic sign of overfitting. Dropout(0.3) partially mitigates this, but the gap between train (98.77%) and validation (92.18%) accuracy at epoch 20 confirms the model has overfit the training data.

---

## Requirements

```
tensorflow
numpy
matplotlib
```

Install all dependencies:

```bash
pip install tensorflow numpy matplotlib
```

---

## Usage

1. Open `Lab_assessment1.ipynb` in Jupyter Notebook or Google Colab.
2. Run all cells sequentially — the dataset downloads automatically.

---

## Technologies Used

- **Python 3**
- **TensorFlow / Keras** — CNN architecture, training, evaluation
- **numpy** — array operations and seed setting
- **matplotlib** — training history visualization
