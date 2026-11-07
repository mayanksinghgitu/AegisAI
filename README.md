<<<<<<< HEAD
# 🧠 AI-Based Emotion Detection System (CNN)

An AI-powered facial emotion recognition system developed using **TensorFlow and Keras**.  
This project trains a **Convolutional Neural Network (CNN)** to classify human facial expressions into seven emotion categories using grayscale images.

---

## 📌 Project Overview

The AI-Based Emotion Detection System is designed to automatically detect and classify emotions from facial images.  
It uses a deep learning-based CNN architecture optimized with **data augmentation, batch normalization, and global average pooling** to improve accuracy and reduce overfitting.

This model can be used for **mental health monitoring, patient care, stress detection, and human-computer interaction** applications.

---

## 🎭 Emotion Classes

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

Total Classes: **7**

---

## 🛠️ Libraries & Tools Used

- **Python 3.x**
- **TensorFlow**
- **Keras**
- **NumPy**
- **OS (File Handling)**
- **TensorFlow Dataset API**

---

## 📂 Dataset Details

- Dataset Type: Directory-based image dataset  
- Color Mode: Grayscale  
- Image Format: `.jpg / .png`  
- Dataset Split:
  - **80% Training**
  - **20% Validation**

### Dataset Structure
train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/

---

## 🖼️ Image & Data Configuration

- Image Size: **100 × 100**
- Image Shape: **(100, 100, 1)**
- Batch Size: **64**
- Data Type: `float32`
- Normalization: Pixel values scaled to **[0, 1]**

---

## 🔄 Data Preprocessing & Augmentation

### Preprocessing
- Conversion to float32
- Pixel normalization (`/255.0`)
- Dataset caching and prefetching for performance

### Data Augmentation (Training Only)
- Random Horizontal Flip
- Random Rotation (±5%)
- Random Zoom (±10%)

---

## 🧠 Model Architecture

- Input Layer: `(100, 100, 1)`
- Convolution Block 1: 32 filters + BatchNorm + ReLU + MaxPooling
- Convolution Block 2: 64 filters + BatchNorm + ReLU + MaxPooling
- Convolution Block 3: 128 filters + BatchNorm + ReLU + MaxPooling
- Convolution Block 4: 256 filters + BatchNorm + ReLU + MaxPooling
- Global Average Pooling (instead of Flatten)
- Dense Layer: 256 units (ReLU)
- Dropout: 0.5
- Output Layer: Softmax (7 classes)

---

## ⚙️ Training Configuration

- Optimizer: **Adam**
- Learning Rate: **0.001**
- Loss Function: **Sparse Categorical Crossentropy**
- Metrics: **Accuracy**
- Epochs: **40**

---

## ⏱️ Callbacks Used

- **ModelCheckpoint**
  - Saves best model based on validation accuracy
- **EarlyStopping**
  - Stops training if validation loss does not improve (patience = 7)
- **ReduceLROnPlateau**
  - Reduces learning rate when validation loss plateaus

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
- pip install tensorflow
- pip install numpy
- pip install pandas

=======
# AegisAI
ADAPTIVE MULTIMODAL AI FOR PERSONALIZED HEALTHCARE &amp; WELLNESS RISK ASSESSMENT
>>>>>>> origin/main
