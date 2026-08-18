# Iris Flower Classification

## Project Overview

This project implements a machine learning solution to classify iris flowers into three species:
- **Setosa**
- **Versicolor**
- **Virginica**

The project demonstrates a complete machine learning workflow from data loading, exploration, preparation, model training, to evaluation and comparison.

## Dataset

The Iris dataset contains 150 samples of iris flowers with 4 measurements (features):
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

**Dataset Statistics:**
- Total Samples: 150
- Training Samples: 120 (80%)
- Testing Samples: 30 (20%)
- Classes: 3 (Setosa, Versicolor, Virginica)
- Samples per class: 50

## Project Structure

```
EXPS_Task1_Iris_Classification/
├── iris_classification.ipynb      # Main Jupyter notebook
├── results/                        # Output visualizations
│   ├── 01_feature_distributions.png
│   ├── 02_model_comparison.png
│   └── 03_confusion_matrices.png
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── LICENSE                         # Project license
```

## Models Used

We trained and compared 3 different classification algorithms:

### 1. K-Nearest Neighbors (KNN, k=3)
- **Principle:** Classifies based on 3 nearest neighbors in training data
- **How it works:** For a new flower, find the 3 most similar flowers and vote on species
- **Advantage:** Simple and intuitive
- **Note:** Requires feature scaling for best results

### 2. Logistic Regression
- **Principle:** Finds linear decision boundaries between species
- **How it works:** Learns a linear combination of features to separate classes
- **Advantage:** Fast and interpretable
- **Best for:** Linearly separable data

### 3. Support Vector Machine (SVM)
- **Principle:** Finds optimal hyperplane to separate classes
- **How it works:** Uses kernel trick to find non-linear boundaries
- **Advantage:** Powerful for complex patterns
- **Best for:** Non-linear separation

## Results

All three models achieved high accuracy on the test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| KNN | ~96% | ~96% | ~96% | ~96% |
| Logistic Regression | ~97% | ~97% | ~97% | ~97% |
| SVM | ~98% | ~98% | ~98% | ~98% |

*Note: Exact percentages may vary due to randomness in train-test split*

## Key Findings

1. **Setosa Separation:** Setosa is easily separable from the other two species due to smaller petal measurements
2. **Versicolor vs Virginica:** These two species have overlapping feature ranges, making them harder to distinguish
3. **Feature Importance:** Petal measurements appear more discriminative than sepal measurements
4. **Scaling Impact:** Feature scaling significantly improved KNN and SVM performance

## Data Preparation

### Train-Test Split (80-20)
- 80% of data used for training models
- 20% of data used for testing/evaluation
- Stratification ensures balanced class distribution in both sets

### Feature Scaling
Feature scaling is crucial for distance-based algorithms (KNN, SVM):

**Why scale?**
- Raw features have different ranges (e.g., Sepal Length: 4-8, Petal Length: 1-7)
- Distance-based algorithms would be dominated by larger values
- Scaling ensures all features contribute equally

**How we scale:**
- StandardScaler: X_scaled = (X - mean) / standard_deviation
- After scaling: All features have mean ≈ 0 and std ≈ 1

**Example:**
- Before: Sepal Length ranges 4-8, Petal Width ranges 0.1-2.5
- After: Both range approximately -2 to +2 on standardized scale

## Visualizations

### 1. Feature Distributions (01_feature_distributions.png)
Shows how each measurement varies across the three iris species.
- **Red histogram:** Setosa
- **Teal histogram:** Versicolor
- **Blue histogram:** Virginica

**Interpretation:** We can see which features best separate the species.

### 2. Model Comparison (02_model_comparison.png)
Bar chart comparing accuracy of all three models.
- Shows which model performs best
- Easy to compare performance at a glance

### 3. Confusion Matrices (03_confusion_matrices.png)
Shows predictions vs actual labels for each model.
- **Rows:** Actual species
- **Columns:** Predicted species
- **Diagonal:** Correct predictions (dark blue)
- **Off-diagonal:** Mistakes (light blue)

**Interpretation:** If two species are frequently confused, their off-diagonal values will be high.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EXPS_Task1_Iris_Classification.git
cd EXPS_Task1_Iris_Classification
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `iris_classification.ipynb`

3. Run all cells to:
   - Load and explore the dataset
   - Prepare data (train-test split and scaling)
   - Train all three models
   - Evaluate model performance
   - Generate visualizations

### Running as Python Script

To run as a Python script instead of Jupyter:
```bash
python iris_classification.py
```

## Dependencies

See `requirements.txt` for complete list. Main packages:
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computing
- **scikit-learn:** Machine learning algorithms
- **matplotlib:** Data visualization
- **seaborn:** Statistical data visualization

## Conclusions

1. **Best Model:** SVM achieved the highest accuracy (~98%)
2. **Simplicity vs Performance:** KNN is simpler but SVM performs better
3. **Practical Use:** All three models are suitable for iris classification
4. **Feature Scaling:** Essential for distance-based algorithms

## Internship Requirements Met

✓ Dataset loaded and explored  
✓ Data prepared with train-test split  
✓ Feature scaling implemented  
✓ Multiple classification models trained  
✓ Models evaluated with proper metrics  
✓ Visualizations created and explained  
✓ Results documented and interpreted  
✓ Code is clean and well-documented  

## Author

Created for EXPS Nexus Data Science Internship

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

EXPS Nexus
- Website: www.exps.tech
- Email: hr@expsdz.com
- WhatsApp: +213 551 283 656

## References

- Iris Dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
- Scikit-learn Documentation: https://scikit-learn.org/
- Machine Learning Basics: https://en.wikipedia.org/wiki/Machine_learning
