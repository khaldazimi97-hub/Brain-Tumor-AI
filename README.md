# 🧠 Brain Tumor MRI Classification <br/>_EfficientNetV2 + Test-Time Augmentation = High-Accuracy Medical Imaging_

[![View Notebook](https://img.shields.io/badge/-View%20Notebook-blue?logo=jupyter)](./) <!-- TODO: Link to your notebook if public -->
[![Report Bug](https://img.shields.io/badge/-Report%20Bug-red?logo=github)](../../issues)

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Methodology & Architecture](#-methodology--architecture)
- [Dataset & EDA](#-dataset--eda)
- [Training Strategy](#-training-strategy)
- [Results & Evaluation](#-results--evaluation)
- [Tech Stack](#-tech-stack)

- [Future Work](#-future-work)

---

## 🌟 Overview
Manual brain tumor detection is time-consuming and prone to error. This project presents a deep learning pipeline using EfficientNetV2B3 to automatically classify brain MRI scans into four categories with **94.94% accuracy**.

- **Four Classes:** Glioma, Meningioma, Pituitary Tumors, Healthy Brain
- **Approaches Used:** Transfer learning, optimized data pipeline, test-time augmentation

---

## 🏗️ Methodology & Architecture

### 1. Data Pipeline (`tf.data`)
- **Modern TensorFlow `tf.data` API:** Replaces legacy `ImageDataGenerator`
- **In-graph Augmentation:** GPU-accelerated (RandomFlip, RandomRotation, RandomZoom)
- **Prefetching:** Overlaps data preprocessing and model execution

### 2. Model Backbone
- **EfficientNetV2B3:** State-of-the-art tradeoff between parameter efficiency and feature extraction
- **Batch Normalization Fine-Tuning:** Adapts features to MRI pixel statistics

### 3. Test-Time Augmentation (TTA)
- **Robust Inference:** Predicts each image in 3 ways—original, horizontally flipped, vertically flipped—and averages confidence scores to smooth out edge-case misclassifications

---

## 📊 Dataset & EDA

- **Dataset:** Brain Tumor MRI Dataset
- **Total Images:** 7,023
- **Training Set:** 5,600 (1,400 per class)
- **Testing Set:** 1,600 (400 per class)
- **Balance:** Perfectly balanced—no oversampling/class weighting required

**Training Curves:**  
_Validation accuracy and loss indicate stable convergence without severe overfitting._

---

## ⚔️ Training Strategy

| Phase    | Action                                   | Learning Rate | Epochs |
|----------|------------------------------------------|---------------|--------|
| Phase 1  | Freeze backbone, train head              | 1e-3          | 10     |
| Phase 2  | Unfreeze top 30 layers, fine-tune        | 5e-5          | 25     |

- **Early Stopping** and **ReduceLROnPlateau** to prevent overfitting and automate learning rate adjustments.

---

## 🏆 Results & Evaluation

- **Validation Accuracy:** 94.94% (with TTA)
- **Confusion Matrix:** High diagonal values (almost all predictions correct)
- **Per-Class Performance:**
  - **No Tumor:** 100.00% (0 False Positives)
  - **Pituitary:** 100.00% (0 False Positives)
  - **Meningioma:** 93.00%
  - **Glioma:** 86.75%  
    ⮡ _Most confusion between Glioma and Meningioma—an acknowledged challenge in literature_

---

## 💻 Tech Stack

- **Language:** Python 3.12
- **Deep Learning:** TensorFlow 2.15 / Keras
- **Data Handling:** NumPy, OpenCV
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** scikit-learn
- **Environment:** Google Colab (T4 GPU)

---




---
## 🔮 Future Work

- **Explainability:** Grad-CAM heatmaps to show model focus regions for medical transparency
- **Ensemble Models:** Combine EfficientNet with Vision Transformers for greater reliability
- **Web Deployment:** FastAPI + React web app for real-world clinical testing and accessibility

---

**Built by [Khalid Azimi](https://github.com/khaldazimi97-hub) | Computer Science Applicant**  
_Contributions welcome!_
