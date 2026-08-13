"""
Iris Flower Classification Model
Trains multiple ML algorithms for iris species classification
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class IrisClassifier:
    """Iris flower classification using multiple algorithms"""
    
    def __init__(self, random_state=42):
        """Initialize the classifier"""
        self.random_state = random_state
        self.iris = load_iris()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        print("✓ IrisClassifier initialized")
        
    def load_and_split_data(self, test_size=0.25):
        """Load iris dataset and split into train/test sets"""
        X = self.iris.data
        y = self.iris.target
        
        # Split: 75% train, 25% test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Standardize features (important for KNN and Logistic Regression)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Data loaded: {len(self.X_train)} training, {len(self.X_test)} test samples")
        print(f"  Dataset: {len(self.iris.target_names)} classes, {self.iris.data.shape[1]} features")
        
    def train_models(self):
        """Train all classification models"""
        self.models = {
            'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(max_iter=200, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            print(f"✓ Trained: {name}")
    
    def evaluate_all(self):
        """Evaluate all models and return results DataFrame"""
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            result = {
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, average='weighted'),
                'Recall': recall_score(self.y_test, y_pred, average='weighted'),
                'F1-Score': f1_score(self.y_test, y_pred, average='weighted')
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_best_model(self):
        """Return the best performing model"""
        results_df = self.evaluate_all()
        best_idx = results_df['Accuracy'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        return best_model_name, self.models[best_model_name]
    
    def predict_species(self, measurements):
        """
        Predict iris species from flower measurements
        
        Args:
            measurements: [sepal_length, sepal_width, petal_length, petal_width]
        
        Returns:
            Predicted species name
        """
        best_name, best_model = self.get_best_model()
        scaled_measurements = self.scaler.transform([measurements])
        prediction_idx = best_model.predict(scaled_measurements)[0]
        return self.iris.target_names[prediction_idx]
    
    def get_confusion_matrix(self, model_name=None):
        """Get confusion matrix for a specific model"""
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)
    
    def plot_results(self, save_path=None):
        """Plot model comparison and confusion matrix"""
        results_df = self.evaluate_all()
        
        # Model comparison bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
            kind='bar', ax=axes[0], color=['#185FA5', '#0F6E56', '#993C1D', '#534AB7']
        )
        axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Model')
        axes[0].set_ylim([0.85, 1.0])
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Confusion matrix for best model
        best_name, best_model = self.get_best_model()
        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   xticklabels=self.iris.target_names,
                   yticklabels=self.iris.target_names)
        axes[1].set_title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize
    clf = IrisClassifier(random_state=42)
    
    # Load and prepare data
    clf.load_and_split_data(test_size=0.25)
    
    # Train models
    clf.train_models()
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    results = clf.evaluate_all()
    print(results.to_string(index=False))
    
    # Get best model
    best_name, _ = clf.get_best_model()
    print(f"\n✓ Best model: {best_name}")
    
    # Make a prediction
    test_measurement = [5.1, 3.5, 1.4, 0.2]  # Setosa
    prediction = clf.predict_species(test_measurement)
    print(f"\n✓ Predicted species for {test_measurement}: {prediction}")
    
    # Plot results
    print("\nGenerating visualizations...")
    clf.plot_results(save_path='results.png')