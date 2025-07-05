# Deep Learning for Image Classification  
**University Project (Universidade do Minho)**

## 📌 Overview
This project explores deep learning techniques for image classification across three datasets of increasing complexity: **CIFAR-10**, **CIFAR-100**, and **Stanford Dogs**. A variety of neural network architectures were implemented, including:

- Multi-Layer Perceptrons (MLP)
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory networks (LSTM)
- Vision Transformers (ViT)
- Transfer Learning with InceptionV3, ResNet50, and ViT-B16-FE

The objective was to evaluate performance across these datasets and models, optimizing via hyperparameter tuning and data preprocessing techniques.

---

## 📂 Datasets
1. **CIFAR-10**  
   - 10 classes (animals & vehicles)  
   - 32×32 resolution images  
   - Balanced dataset

2. **CIFAR-100**  
   - 100 fine-grained classes grouped into 20 superclasses  
   - 32×32 resolution  
   - More complex and less separable than CIFAR-10

3. **Stanford Dogs**  
   - 120 dog breeds  
   - High-resolution images (224×224)  
   - Real-world class imbalance and visual similarity

---

## 🧠 Architectures
- **MLP:** Simple 3-layer model, used as a performance baseline  
- **CNN:** Custom models with/without data augmentation (horizontal flip, zoom, rotation)  
- **LSTM:** Applied to reshaped image data for sequence modeling  
- **Vision Transformer (ViT):** Patch-based transformer encoder  
- **Transfer Learning:**  
  - `InceptionV3`  
  - `ResNet50`  
  - `ViT-B16-FE` (via `tensorflow_hub`)

---

## 🧪 Optimization
- **Learning Rate Scheduling**
- **Hyperparameter Tuning:**  
  - Grid Search  
  - Random Search  
  - Hyperband (preferred due to efficiency)
- **Cross-Validation:**  
  Used 5-fold cross-validation for performance validation, especially on CNN and InceptionV3.

---

## ⚙️ Setup & Usage

### 📦 Dependencies
Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🚀 Running the Project

```bash
# Train a specific model
python train.py --model cnn --dataset cifar10

# Evaluate model
python evaluate.py --model cnn --dataset cifar10
```

> Ensure appropriate folders and datasets are downloaded before running.

---

## 📊 Model Performance Comparison

| Model         | CIFAR-10 Accuracy | CIFAR-100 Accuracy | Stanford Dogs Accuracy |
|---------------|-------------------|---------------------|------------------------|
| MLP           | 53.5%             | 25.4%               | 4.1%                   |
| CNN           | **75.2%**         | **44.1%**           | 21.1%                  |
| CNN + DA      | 70.0%             | 37.8%               | –                      |
| LSTM          | 56.8%             | 29.0%               | Not attempted          |
| ViT           | 65.3%             | 36.6%               | 86.1%                  |
| InceptionV3   | –                 | –                   | **86.8%**              |
| ResNet50      | –                 | –                   | 65.4%                  |

Notes: ViT with Kaggle API was used only in SDD (Stanford Dogs dataset)

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Google Colab
- Kaggle API
- TensorFlow Hub

---