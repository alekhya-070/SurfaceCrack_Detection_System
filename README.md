# Automated Surface Crack Detection for Quality Control

> End-to-end deep learning system for detecting surface cracks using transfer learning with pretrained CNN architectures (ResNet50, InceptionV3, VGG16). Designed for automated defect detection in construction materials.

---

## 🧩 Overview

This project implements an automated crack detection pipeline for quality control applications. Using transfer learning on pretrained models, it identifies surface cracks from high-resolution images of concrete and other construction materials.

The system was trained on **25,000 labeled crack images**, achieving **99.9% accuracy** with ResNet50.

---

## 🚀 Key Features

* Comparative analysis of **ResNet50**, **InceptionV3**, and **VGG16**.
* Transfer learning with frozen convolutional layers and fine-tuning.
* Image preprocessing using **OpenCV** for noise removal and contrast enhancement.
* Data augmentation to improve generalization.
* Model evaluation (accuracy, precision, recall, F1-score, confusion matrix).
* TensorFlow-based training and inference pipeline.
* Visual output overlay highlighting detected crack regions.

---

## 🧱 Tech Stack

* **Language:** Python 3.8+
* **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Scikit-learn

---

## 📊 Dataset

* **Total Images:** 25,000
* **Classes:** Crack / No Crack
* **Split:** 70% train, 20% validation, 10% test

Preprocessing includes:

* Resizing to 224×224 (compatible with CNN input)
* Histogram equalization for contrast
* Random rotation, flip, and zoom augmentations

---

## 🧠 Training

Train using ResNet50 as baseline:

```bash
python src/train.py --model resnet50 --epochs 20 --batch-size 32 --lr 0.0001
```

Training steps:

1. Load pretrained CNN (ImageNet weights)
2. Replace final classification layer (Dense → 2 classes)
3. Freeze early layers; fine-tune last convolutional blocks
4. Train on crack dataset

---

## 📈 Evaluation

Evaluate trained models:

```bash
python src/evaluate.py --model models/resnet50_model.h5 --data data/test/
```

Outputs:

* Accuracy, Precision, Recall, F1-Score
* Confusion matrix visualization
* Per-class performance summary

**Results:**

| Model    | Accuracy  | Precision | Recall | F1-Score |
| -------- | --------- | --------- | ------ | -------- |
| ResNet50 | **99.9%** | 99.8%     | 99.9%  |          |
