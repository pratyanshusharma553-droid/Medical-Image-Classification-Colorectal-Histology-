# 🧠 Medical Image Classification with CNNs

> Deep learning project exploring custom convolutional networks and transfer learning for multi-class histology image classification.

---

## 🌟 **PROJECT HIGHLIGHTS**

- 🚀 Built and trained a **custom CNN architecture** from scratch  
- 🔁 Applied **transfer learning** using a pretrained ResNet50 model  
- 🧩 Used **Global Average Pooling** to reduce overfitting and model size  
- 🎯 Performed **fine-tuning** to adapt pretrained features to a new domain  
- 📊 Evaluated models using multiple **classification metrics**  
- 📈 Visualized performance with **confusion matrices and training curves**

---

## 📂 **DATASET OVERVIEW**

| Property | Value |
|---------|-------|
| Categories | 8 tissue classes |
| Total Images | ~5,000 |
| Train/Test Split | 90% / 10% |
| Image Size | 224 × 224 × 3 |

---

## 🏗 **MODEL APPROACHES**

### 🔹 Custom Convolutional Neural Network
- Convolution + pooling layers  
- Global Average Pooling  
- Softmax classification head  

### 🔹 Transfer Learning Model
- Pretrained ResNet50 backbone  
- Initial feature extraction (frozen layers)  
- Fine-tuning of higher-level layers  
- GAP + Dense classifier  

---

## 📊 **PERFORMANCE SUMMARY**

| Model | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| 🧠 Custom CNN | **0.73** | 0.75 | 0.73 | 0.72 |
| 🔁 ResNet50 (Fine-tuned) | 0.55 | 0.59 | 0.55 | 0.52 |

---

## 📈 **EVALUATION STRATEGY**

✔ Accuracy  
✔ Macro Precision  
✔ Macro Recall  
✔ Macro F1-score  
✔ Confusion Matrix  

---

## 🛠 **TECH STACK**

- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- TensorFlow Datasets  

---

## ▶️ **HOW TO RUN**

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
