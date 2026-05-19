# 🧠 Neural Networks Benchmarking  
## CNN vs FFNN for Brain Tumor Classification

A complete deep learning benchmarking project comparing:

- Feedforward Neural Networks (FFNN)
- Convolutional Neural Networks (CNN)

on a Brain MRI Classification dataset using TensorFlow & Streamlit.

---

# 📌 Project Overview

This project investigates the trade-off between:

- Model Accuracy
- Generalization Ability
- Training Time
- Inference Speed
- Overfitting Behavior

The models were trained completely from scratch without using any pre-trained weights.

---

# 🧪 Project Objectives

- Perform Exploratory Data Analysis (EDA)
- Build preprocessing pipelines
- Design FFNN and CNN architectures
- Compare performance using multiple metrics
- Analyze overfitting and regularization
- Conduct optimization experiments
- Deploy an interactive Streamlit dashboard

---

# 🗂️ Dataset Structure

```bash
project/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Plotly
- Streamlit
- Scikit-learn
- Pillow
- gdown

---

# 📊 Exploratory Data Analysis (EDA)

The project includes:

- Dataset summary
- Class distribution visualization
- Sample MRI image visualization
- Duplicate checking
- Dataset balancing inspection

---

# 🧹 Preprocessing Pipeline

The preprocessing pipeline includes:

- Image resizing → `128 × 128`
- Pixel normalization using Rescaling
- Train / Validation split
- Dataset caching & prefetching
- Optional Data Augmentation

---

# 🧠 Model Architectures

## 🔹 FFNN Architecture

```python
Sequential([
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])
```

### Characteristics
- Faster training
- Simpler architecture
- Prone to overfitting on image tasks
- Lacks spatial feature extraction

---

## 🔹 CNN Architecture

```python
Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(num_classes, activation='softmax')
])
```

### Characteristics
- Learns spatial features
- Better generalization
- Higher classification accuracy
- More computationally expensive

---

# 📈 Evaluation Metrics

The following metrics were used:

- Accuracy
- Loss
- F1-Score
- Training Time
- Inference Speed

---

# 🔬 Optimization Experiments

## CNN Experiments

- High Learning Rate
- No Dropout
- Low Learning Rate
- Data Augmentation

## FFNN Experiments

- High Learning Rate
- No Dropout
- Low Learning Rate
- Deeper Architecture

---

# 📉 Key Findings

✅ CNN significantly outperformed FFNN on image classification tasks.

✅ Dropout reduced overfitting and improved generalization.

✅ Lower learning rates produced more stable training.

✅ Data augmentation improved CNN robustness.

---

# 🚀 Streamlit Dashboard Features

The interactive dashboard includes:

- Project Overview
- EDA Visualization
- Architecture Comparison
- Benchmarking Results
- Optimization Experiments
- Live MRI Upload & Prediction Simulation

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🌐 Streamlit Deployment

This project is deployed using Streamlit Community Cloud.

🔗 Live Demo:  
https://brain-tumor-classification-engymohamedhanafy.streamlit.app/
---

# 📚 Academic Integrity

- All models were trained from scratch
- No pre-trained weights were used
- All experiments were manually implemented and analyzed

---

# 👩‍💻 Author

Developed as part of a Neural Networks benchmarking project focused on CNN vs FFNN performance analysis for Brain Tumor MRI Classification.
