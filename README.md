# 🔬 Skin Cancer Detection with CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-82%25-brightgreen.svg)

**Fast & efficient CNN for classifying skin lesions into 7 categories**

[Quick Start](#-quick-start) • [Results](#-results) • [Dataset](#-dataset)

<img src="https://raw.githubusercontent.com/yourusername/repo/main/images/banner.png" alt="Skin Lesion Detection" width="700"/>

</div>

---

## 🎯 Overview

Lightweight CNN model for automated skin lesion classification using the **HAM10000 dataset**. Optimized for CPU with **82% accuracy** in just **35 minutes** training time.

### 📊 7 Diagnostic Categories

<div align="center">

| Disease | Code | % |
|---------|------|---|
| Melanocytic nevi | nv | 67% |
| Melanoma | mel | 11% |
| Benign keratosis | bkl | 11% |
| Basal cell carcinoma | bcc | 5% |
| Actinic keratoses | akiec | 3% |
| Vascular lesions | vasc | 1% |
| Dermatofibroma | df | 1% |

</div>

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/skin-cancer-detection.git
cd skin-cancer-detection

# Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

# Download dataset from Kaggle
# Place hmnist_28_28_RGB.csv in data/ folder

# Train model
python train.py
```

---

## ✨ Features

- ⚡ **Fast**: 35 min training on CPU
- 🎯 **Accurate**: 82% validation accuracy  
- 💾 **Lightweight**: 5MB model size
- 🔄 **Data Augmentation**: Handles class imbalance
- 📊 **Visualizations**: Training plots & confusion matrix

---

## 🏗️ Model Architecture

<div align="center">

```
Input (28×28×3)
    ↓
Conv2D(32) → Conv2D(32) → MaxPool → Dropout
    ↓
Conv2D(64) → Conv2D(64) → MaxPool → Dropout
    ↓
Conv2D(128) → MaxPool → Dropout
    ↓
Dense(256) → Dense(128) → Output(7)
```

**Total Parameters:** 468K

</div>

---

## 📈 Results

<div align="center">

### Training Performance

<img src="https://raw.githubusercontent.com/yourusername/repo/main/images/training_history.png" alt="Training History" width="650"/>

### Confusion Matrix

<img src="https://raw.githubusercontent.com/yourusername/repo/main/images/confusion_matrix.png" alt="Confusion Matrix" width="500"/>

</div>

### Per-Class Accuracy

```
Class      Accuracy    Samples
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
akiec      68.3%       82
bcc        76.9%       104
bkl        78.2%       219
df         54.5%       22
mel        70.0%       220
nv         91.3%       1341
vasc       79.0%       19
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall    82.5%       2007
```

---

## 💻 Usage

### Training

```python
python train.py
```

### Prediction

```python
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('ham10000_model.h5')

# Predict
prediction = model.predict(image)
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
result = classes[np.argmax(prediction)]
print(f"Prediction: {result}")
```

---

## 📊 Dataset

**HAM10000** - 10,015 dermatoscopic images at 28×28 RGB

- 🔬 53% histopathologically confirmed
- 🌍 Multi-source collection
- ⚖️ CC BY-NC-SA 4.0 License

**Source:** [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---



## 🤝 Contributing

Contributions welcome! Fork, create a branch, commit, and open a PR.

---

## 📄 License

MIT License • Dataset: CC BY-NC-SA 4.0

---

## 🙏 Credits

- **Dataset:** [HAM10000](https://doi.org/10.1038/sdata.2018.161) by Tschandl et al.
- **Platform:** [Kaggle](https://www.kaggle.com/)

---

<div align="center">

### ⭐ Star if you find this useful!

**Made with ❤️ for medical AI**

</div>
