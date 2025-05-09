import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import os

# Create directories for results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data...")
X_train = np.load('models/X_train_resampled.npy')
y_train = np.load('models/y_train_resampled.npy')
X_val = np.load('models/X_val_scaled.npy')
y_val = np.load('models/y_val.npy')
X_test = np.load('models/X_test_scaled.npy')
y_test = np.load('models/y_test.npy')

# Define models to train
models = {
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Define a function for evaluating models
def evaluate_model(model, X, y, model_name, dataset_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    # Create and save confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Phishing'], 
                yticklabels=['Safe', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
    plt.savefig(f'results/{model_name}_{dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create and save ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} ({dataset_name})')
    plt.legend()
    plt.savefig(f'results/{model_name}_{dataset_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# Results storage
results = {
    'model': [],
    'train_time': [],
    'val_accuracy': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': [],
    'val_auc': [],
    'test_accuracy': [],
    'test_precision': [],
    'test_recall': [],
    'test_f1': [],
    'test_auc': []
}

# Train and evaluate each model
print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train with cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    start_time = time.time()
    
    # For simplicity, train on the whole training set instead of CV folds
    # But we'll still use CV for hyperparameter tuning in a real scenario
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save the model
    with open(f'models/{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to models/{name.lower().replace(' ', '_')}_model.pkl")
    
    # Evaluate on validation set
    print(f"Evaluating {name} on validation set...")
    val_metrics = evaluate_model(model, X_val, y_val, name, 'validation')
    
    # Evaluate on test set
    print(f"Evaluating {name} on test set...")
    test_metrics = evaluate_model(model, X_test, y_test, name, 'test')
    
    # Store results
    results['model'].append(name)
    results['train_time'].append(train_time)
    results['val_accuracy'].append(val_metrics['accuracy'])
    results['val_precision'].append(val_metrics['precision'])
    results['val_recall'].append(val_metrics['recall'])
    results['val_f1'].append(val_metrics['f1'])
    results['val_auc'].append(val_metrics['auc'])
    results['test_accuracy'].append(test_metrics['accuracy'])
    results['test_precision'].append(test_metrics['precision'])
    results['test_recall'].append(test_metrics['recall'])
    results['test_f1'].append(test_metrics['f1'])
    results['test_auc'].append(test_metrics['auc'])
    
    print(f"{name} evaluation completed.")
    print(f"Validation: Accuracy={val_metrics['accuracy']:.4f}, Precision={val_metrics['precision']:.4f}, "
          f"Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")
    print(f"Test: Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, "
          f"Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")

# Create results DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/model_comparison.csv', index=False)
print("\nResults saved to results/model_comparison.csv")

# Visualize model comparison
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
for metric in metrics:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results['model']))
    width = 0.35
    
    plt.bar(x - width/2, results_df[f'val_{metric}'], width, label='Validation')
    plt.bar(x + width/2, results_df[f'test_{metric}'], width, label='Test')
    
    plt.xlabel('Models')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xticks(x, results_df['model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()
    

print("\nModel comparison visualizations saved to results/ directory")
print("\nTraining and evaluation complete!")