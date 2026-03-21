# 🧬 OncoScan  
### Breast Cancer Risk Assessment System  

🔗 **Live Demo:**  
https://breast-cancer-oncoscangit.streamlit.app  

An AI-powered clinical decision support tool for breast cancer malignancy prediction.

---

## 📌 Overview

**OncoScan** analyzes tumor morphology measurements derived from Fine Needle Aspirate (FNA) imaging and predicts malignancy probability using a trained Support Vector Machine (SVM) classifier.

The system:
- Accepts nuclear morphology features as input  
- Outputs:
  - Malignancy **risk percentage**
  - **Diagnostic verdict** (Benign / Malignant)
  - **Confidence level**
- Displays results via a responsive, dark-themed clinical UI  

---

## ✨ Features

- **🤖 AI-Powered Prediction**  
  SVM classifier (RBF kernel) trained on the UCI Breast Cancer Wisconsin dataset  

- **🧠 Correlation-Based Feature Selection**  
  Removes highly correlated features (> 0.90), reducing:
  - **30 → 20 features**

- **📊 Structured Input Interface**  
  Three grouped sections:
  - Mean features  
  - Standard Error features  
  - Worst features  

- **📈 Risk Score Gauge**  
  Visual malignancy probability indicator (0–100%)

- **🎯 Confidence Metrics**  
  Displays benign probability and prediction confidence  

- **⚖️ Borderline Detection**  
  Flags uncertain predictions (45–55%)  

- **🧬 Animated Visualization**  
  SVG-based animated cell with scanning effects  

- **📱 Fully Responsive Design**  
  Works seamlessly across devices  

- **🔄 Dynamic Feature Loading**  
  Loads features from `feature_columns.pkl` (no hardcoding)

---

## ⚙️ How It Works

Raw CSV (569 samples, 30 features)
↓
Drop columns (id, Unnamed: 32)
↓
Encode target: B → 0, M → 1
↓
Train/Test Split (80/20, stratified)
↓
Correlation-Based Feature Selection
(drop features > 0.90 correlation)
↓
20 features retained
↓
Model Comparison:
Logistic Regression | SVM | Random Forest
↓
SVM selected (best ROC-AUC)
↓
Pipeline:
StandardScaler → SVM (RBF kernel)
↓
Saved:
model.pkl | scaler.pkl | feature_columns.pkl


---

## 📊 Model Performance

| Metric | Score |
|------|------|
| Algorithm | SVM (RBF Kernel) |
| Cross-Validation ROC-AUC | **98.2%** |
| Test ROC-AUC | **99.4%** |
| Class Weight | `balanced` |

- Dataset distribution:
  - **357 Benign**
  - **212 Malignant**

---

## 📁 Dataset

| Property | Value |
|--------|------|
| Source | UCI Breast Cancer Wisconsin (Diagnostic) |
| Total Samples | 569 |
| Benign | 357 |
| Malignant | 212 |
| Original Features | 30 |
| Selected Features | 20 |

---

## 🧰 Tech Stack

| Component | Technology |
|----------|-----------|
| Frontend | Streamlit |
| ML Model | Scikit-learn (SVM) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Persistence | Joblib |
| Fonts | Exo 2, JetBrains Mono |

---

## 📂 Project Structure
├── app.py
├── trained_model.py
├── model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── removed_columns.pkl
├── Breast_Cancer.csv
└── requirements.txt


## 📥 Input Features

Features are dynamically loaded from `feature_columns.pkl`.

### Mean Features (7)
- radius  
- texture  
- smoothness  
- compactness  
- concavity  
- symmetry  
- fractal dimension  

### Standard Error Features (8)
- radius  
- texture  
- smoothness  
- compactness  
- concavity  
- concave points  
- symmetry  
- fractal dimension  

### Worst Features (5)
- smoothness  
- compactness  
- concavity  
- symmetry  
- fractal dimension  






