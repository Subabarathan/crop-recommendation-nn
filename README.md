# 🌾 Crop Recommendation System using Neural Networks

A machine learning project that recommends the most suitable crop to grow based on soil nutrient levels and climate conditions, using a feedforward neural network built with TensorFlow/Keras.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Sample Prediction](#sample-prediction)

---

## Overview

Given seven input parameters — soil nutrients (N, P, K), temperature, humidity, pH, and rainfall — this model predicts the best crop to cultivate from **22 possible crops**. The model uses a simple but effective multi-layer perceptron (MLP) with dropout regularisation and early stopping.

---

## Dataset

**Source:** [Kaggle – Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

| Feature       | Description                          |
|---------------|--------------------------------------|
| `Nitrogen`    | Nitrogen content ratio in soil       |
| `Phosphorus`  | Phosphorus content ratio in soil     |
| `Potassium`   | Potassium content ratio in soil      |
| `Temperature` | Temperature in °C                    |
| `Humidity`    | Relative humidity in %               |
| `pH_Value`    | pH value of the soil                 |
| `Rainfall`    | Rainfall in mm                       |
| `Crop`        | **Target** — recommended crop label  |

- **Rows:** 2,200
- **Classes:** 22 crops (Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee)

> The dataset file `Crop_Recommendation.csv` is included in this repository. It is also available on [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

---

## Project Structure

```
crop-recommendation-nn/
├── crop_recommendation.ipynb   # Main notebook (EDA → training → evaluation)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/crop-recommendation-nn.git
cd crop-recommendation-nn

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook crop_recommendation.ipynb
```

---

## Usage

Open `crop_recommendation.ipynb` and run all cells in order. The notebook covers:

1. **Import libraries**
2. **Load & explore the dataset** — shape, class distribution, descriptive stats
3. **Preprocessing** — label encoding, feature scaling, train/test split
4. **Build the neural network** — Keras Sequential model
5. **Train the model** — with early stopping on validation loss
6. **Evaluate** — test accuracy, classification report
7. **Plot training history** — accuracy & loss curves
8. **Make a prediction** — single-sample inference

---

## Model Architecture

```
Input (7 features)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.2)
    ↓
Dense(32, ReLU)
    ↓
Dense(22, Softmax)   ← 22 crop classes
```

- **Optimiser:** Adam  
- **Loss:** Sparse Categorical Cross-Entropy  
- **Regularisation:** Dropout + Early Stopping (patience=5)

---

## Results

| Metric         | Score    |
|----------------|----------|
| Test Accuracy  | ~97%     |

> Exact results may vary slightly across runs due to random weight initialisation.

---

## Sample Prediction

```python
sample = np.array([[90, 40, 40, 25.0, 80.0, 6.5, 200.0]])
# → Recommended Crop: Rice  (Confidence: 98.XX%)
```

---

## License

This project is licensed under the MIT License.
