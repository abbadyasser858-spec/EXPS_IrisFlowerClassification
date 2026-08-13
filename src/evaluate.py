"""
Evaluation utilities for Iris classifier
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model_detailed(clf, model_name):
    """Print detailed evaluation metrics for a model"""
    model = clf.models[model_name]
    y_pred = model.predict(clf.X_test)
    
    print(f"\n{'='*60}")
    print(f"DETAILED EVALUATION: {model_name}")
    print(f"{'='*60}")
    print(classification_report(clf.y_test, y_pred, 
                              target_names=clf.iris.target_names))


def plot_feature_importance(clf, save_path=None):
    """Plot feature importance for Random Forest model"""
    rf_model = clf.models['Random Forest']
    importances = rf_model.feature_importances_
    feature_names = clf.iris.feature_names
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances, color='#185FA5')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance - Random Forest Classifier')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
    
    plt.show()