🧠 Colorectal Histology Image Classification using CNNs

This project explores deep learning approaches for multi-class medical image classification using colorectal histology images. It compares a custom-built Convolutional Neural Network (CNN) with a transfer learning model based on ResNet50 to analyze performance, convergence behavior, and model complexity.

📌 Project Overview

The goal is to classify histology images into 8 tissue categories using convolutional neural networks. Two strategies were evaluated:

Custom CNN trained from scratch

Transfer Learning using a pretrained ResNet50 model with fine-tuning

Both models use Global Average Pooling (GAP) to reduce overfitting and improve generalization.

🗂 Dataset

Name: Colorectal Histology

Classes: 8 tissue types

Total Images: 5,000

Train/Test Split: 90% / 10%

Image Size: 224 × 224 × 3

Source: TensorFlow Datasets

🏗 Models Implemented
🔹 Custom CNN

Multiple Conv2D + Pooling layers

Global Average Pooling

Softmax classifier

🔹 Transfer Learning (ResNet50)

Pretrained on ImageNet

Base layers frozen initially

Top layers fine-tuned

GAP + Dense classification head

📊 Results Summary
Model	Accuracy	Precision	Recall	F1 Score
Custom CNN	0.73	0.75	0.73	0.72
ResNet50 (Fine-tuned)	0.55	0.59	0.55	0.52
🔍 Key Observations

The custom CNN achieved higher accuracy on this dataset.

Transfer learning improved after fine-tuning but remained below the custom model.

GAP helped both models generalize by reducing parameter count.

Deeper models required more computation but did not always guarantee better performance.

📈 Evaluation Metrics

Performance was evaluated using:

Accuracy

Precision (macro)

Recall (macro)

F1-score (macro)

Confusion Matrix visualization

🛠 Tech Stack

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

TensorFlow Datasets

▶️ How to Run
pip install -r requirements.txt
jupyter notebook main.ipynb

📌 Future Improvements

Data augmentation for better generalization

Class imbalance handling

Experimenting with other architectures (EfficientNet, DenseNet)

Hyperparameter tuning
