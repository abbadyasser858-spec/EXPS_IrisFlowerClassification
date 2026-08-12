# EXPS_IrisFlowerClassification
Machine learning project to classify iris flower species using KNN, Logistic Regression, and Random Forest. Part of EXPS Nexus Data Science internship

# Iris Flower Classification 🌸

A machine learning project to classify iris flower species using multiple classification algorithms.

## Project Overview

This project is part of the **EXPS Nexus Data Science Internship** (Cohort 01 - Algeria Pilot).

The goal is to build and train multiple machine learning models that can accurately predict the species of an iris flower based on its physical measurements (sepal length, sepal width, petal length, petal width).

### Dataset
- **Source**: Scikit-learn built-in Iris dataset
- **Samples**: 150 iris flower observations
- **Features**: 4 physical measurements
- **Target Classes**: 3 species (Setosa, Versicolor, Virginica)

---

## 🎯 Models Implemented

This project trains and compares **4 different classification algorithms**:

1. **K-Nearest Neighbors (KNN)** - k=3 and k=5
2. **Logistic Regression** - Probabilistic classifier
3. **Random Forest** - Ensemble learning method

Each model is evaluated on accuracy, precision, recall, and F1-score.

---

## 📊 Expected Results

Based on initial experiments:
- **KNN (k=3)**: ~98.2% accuracy
- **KNN (k=5)**: ~96.4% accuracy  
- **Logistic Regression**: ~97.3% accuracy
- **Random Forest**: ~98.2% accuracy

---

## 📁 Project Structure
EXPS_IrisFlowerClassification/
├── README.md
├── requirements.txt
├── src/
│ ├── classifier.py
│ └── evaluate.py
├── notebooks/
│ └── iris_analysis.ipynb
└── data/
├── model_comparison.png
├── feature_importance.png
└── confusion_matrix.png
---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://abbadyasser858/EXPS_IrisFlowerClassification.git
cd EXPS_IrisFlowerClassification
pip install -r requirements.txt
```

### Running the Code

**Option 1: Jupyter Notebook (Recommended)**
```bash
cd notebooks
jupyter notebook iris_analysis.ipynb
```

**Option 2: Use in Python**
```python
from src.classifier import IrisClassifier

clf = IrisClassifier()
clf.load_and_split_data()
clf.train_models()

results = clf.evaluate_all()
print(results)

prediction = clf.predict_species([5.1, 3.5, 1.4, 0.2])
print(f"Predicted: {prediction}")
```

---

## 📈 Visualizations

The project generates:
- Model performance comparison chart
- Feature importance plot
- Confusion matrix

---

## 🔬 Methodology

- **Data Split**: 75% train, 25% test
- **Preprocessing**: Feature standardization
- **Models**: KNN, Logistic Regression, Random Forest
- **Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 👨‍💼 Internship Details

- **Program**: EXPS Nexus Data Science Internship
- **Cohort**: Algeria Pilot - Cohort 01
- **Duration**: August 10 - September 10, 2026
- **Candidate ID**: EXPS_A-105
- **Mentor**: Thamer Niati

---
## 📞 Contact

- **Website**: www.exps.tech
- **Email**: hr@expsdz.com
- **WhatsApp**: +213 551 283 656
  
---
## 📄 License

MIT License
---

**Author**: Yasser Abbad (EXPS_A-105)  
**Last Updated**: August 2026
