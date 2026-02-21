# 🧬 Genomic Variant Clinical Classification System

> An end-to-end machine learning pipeline for classifying cancer-associated genomic variants using clinical evidence text.

---

## 📌 Project Overview

This project implements a production-ready NLP + gradient boosting pipeline for classifying somatic genomic variants into clinically relevant categories based on associated medical literature.

The system ingests structured mutation metadata and unstructured clinical evidence text, performs advanced feature extraction, and trains a multi-class probabilistic classifier optimized using log-loss — a metric appropriate for medical decision systems.

This project demonstrates:

* Applied Natural Language Processing (NLP)
* Multi-class classification modeling
* Feature engineering for biomedical text
* Gradient boosting optimization
* Model persistence and deployment readiness

---

## 🧠 Problem Statement

In oncology and precision medicine, genomic mutations must be interpreted and categorized based on evidence from scientific literature.

Manual interpretation:

* Is time-consuming
* Requires expert knowledge
* Does not scale with growing biomedical publications

This project builds a machine learning system capable of automatically predicting variant classification based on:

* Gene name
* Specific mutation
* Clinical evidence text

---

## 🗂 Dataset

The dataset contains:

| Feature             | Description                        |
| ------------------- | ---------------------------------- |
| `ID`                | Variant identifier                 |
| `Gene`              | Gene symbol                        |
| `Variation`         | Specific mutation                  |
| `Clinical_Evidence` | Associated medical literature text |
| `Class`             | Target classification (1–9)        |

The classification is multi-class (9 categories).

---

## 🏗 Architecture

### 1️⃣ Data Pipeline

* Custom dataset loader
* Text cleaning (medical citation removal, normalization)
* Structured + unstructured feature merging

### 2️⃣ Feature Engineering

* TF-IDF vectorization (word n-grams)
* Biomedical text normalization
* Gene and mutation metadata integration
* Sparse matrix optimization

### 3️⃣ Model

* XGBoost multi-class classifier
* Objective: `multi:softprob`
* Evaluation metric: Multi-class Log Loss
* Stratified train-test split

### 4️⃣ Model Persistence

* Model saved using `joblib`
* Vectorizer saved separately
* Separate inference script for deployment

---

## ⚙️ Technology Stack

* Python 3.x
* Pandas
* Scikit-learn
* XGBoost
* TF-IDF Vectorization
* Joblib (model persistence)

---

## 📊 Model Evaluation

The model is evaluated using:

* Multi-class Log Loss
* Accuracy score

Log-loss is chosen because:

* It penalizes overconfident incorrect predictions
* It is suitable for probabilistic medical classification tasks

---

## 🚀 How To Run

### 1️⃣ Train the Model

```bash
python train.py
```

This will:

* Load and merge data
* Extract TF-IDF features
* Train the XGBoost classifier
* Save the model and vectorizer

Saved files:

```
models/
 ├── xgb_variant_model.pkl
 └── tfidf_vectorizer.pkl
```

---

### 2️⃣ Run Inference

```bash
python predict.py
```

This will:

* Load the trained model
* Clean input text
* Transform using saved TF-IDF vocabulary
* Output predicted class and probabilities

---

## 🔍 Engineering Highlights

### ✔ Clean Modular Design

* Separate loader
* Separate training logic
* Separate inference script

### ✔ Reproducibility

* Fixed random seed
* Stratified splitting
* Saved vectorizer vocabulary

### ✔ Production-Oriented

* No refitting during inference
* Serialized artifacts
* Predict_proba support for decision thresholds

### ✔ Scalable

* Sparse matrix optimization
* Compatible with LightGBM or transformer upgrades
* Ready for API deployment

---

## 🧪 Example Prediction

Input:

```
BRAF V600E mutation associated with melanoma and response to targeted therapy.
```

Output:

```
Predicted Class: 3
Class Probabilities: [0.02, 0.11, 0.63, ...]
```

---

## 🔬 Future Improvements

Planned enhancements include:

* Character-level TF-IDF for mutation pattern capture
* Hyperparameter optimization
* Model stacking
* LightGBM comparison
* Transformer-based biomedical embeddings (BioBERT / SciBERT)
* REST API deployment (FastAPI)
* Docker containerization
* Clinical explainability layer (SHAP values)

---

## 📈 Potential Applications

* Clinical decision support systems
* Precision oncology pipelines
* Automated mutation triage systems
* Biomedical research indexing
* Health-tech AI platforms

---

## 🧑‍💻 Author

Aaron Muliyunda
Medical Student | Software Engineer | AI Systems Builder

Focused on building intelligent healthcare systems that bridge medicine and machine learning.
