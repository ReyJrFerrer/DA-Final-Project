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
    'cv_accuracy': [],
    'cv_precision': [],
    'cv_recall': [],
    'cv_f1': [],
    'cv_auc': [],
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

# Define a function for cross-validation
def cross_validate_model(model, X, y, cv, model_name):
    """Perform cross-validation and return average metrics"""
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    print(f"Performing {cv.n_splits}-fold cross-validation...")
    fold_num = 1
    
    for train_idx, val_idx in cv.split(X, y):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Train the model on this fold
        model_clone = pickle.loads(pickle.dumps(model))  # Create a deep copy of the model
        model_clone.fit(X_fold_train, y_fold_train)
        
        # Get predictions
        y_pred = model_clone.predict(X_fold_val)
        y_pred_proba = model_clone.predict_proba(X_fold_val)[:, 1]
        
        # Calculate metrics
        fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        fold_metrics['precision'].append(precision_score(y_fold_val, y_pred))
        fold_metrics['recall'].append(recall_score(y_fold_val, y_pred))
        fold_metrics['f1'].append(f1_score(y_fold_val, y_pred))
        fold_metrics['auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
        
        print(f"  Fold {fold_num}: Accuracy={fold_metrics['accuracy'][-1]:.4f}, F1={fold_metrics['f1'][-1]:.4f}")
        fold_num += 1
    
    # Calculate average metrics across folds
    avg_metrics = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}
    std_metrics = {metric: np.std(scores) for metric, scores in fold_metrics.items()}
    
    print(f"CV Results for {model_name}:")
    for metric in avg_metrics:
        print(f"  {metric.capitalize()}: {avg_metrics[metric]:.4f} (±{std_metrics[metric]:.4f})")
    
    return avg_metrics

# Train and evaluate each model
print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform cross-validation on training data
    start_time = time.time()
    cv_metrics = cross_validate_model(model, X_train, y_train, skf, name)
    cv_time = time.time() - start_time
    print(f"Cross-validation completed in {cv_time:.2f} seconds")
    
    # Train final model on the whole training set
    print(f"Training final {name} model on full training set...")
    model_final = pickle.loads(pickle.dumps(model))  # Create a deep copy
    start_time = time.time()
    model_final.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save the final model
    with open(f'models/{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model_final, f)
    print(f"Model saved to models/{name.lower().replace(' ', '_')}_model.pkl")
    
    # Evaluate on validation set
    print(f"Evaluating {name} on validation set...")
    val_metrics = evaluate_model(model_final, X_val, y_val, name, 'validation')
    
    # Evaluate on test set
    print(f"Evaluating {name} on test set...")
    test_metrics = evaluate_model(model_final, X_test, y_test, name, 'test')
    
    # Store results
    results['model'].append(name)
    results['train_time'].append(train_time)
    results['cv_accuracy'].append(cv_metrics['accuracy'])
    results['cv_precision'].append(cv_metrics['precision'])
    results['cv_recall'].append(cv_metrics['recall'])
    results['cv_f1'].append(cv_metrics['f1'])
    results['cv_auc'].append(cv_metrics['auc'])
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
    print(f"CV: Accuracy={cv_metrics['accuracy']:.4f}, Precision={cv_metrics['precision']:.4f}, "
          f"Recall={cv_metrics['recall']:.4f}, F1={cv_metrics['f1']:.4f}, AUC={cv_metrics['auc']:.4f}")
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
    plt.figure(figsize=(14, 7))
    x = np.arange(len(results['model']))
    width = 0.25  # Narrower bars to fit three sets
    
    plt.bar(x - width, results_df[f'cv_{metric}'], width, label='Cross-Validation')
    plt.bar(x, results_df[f'val_{metric}'], width, label='Validation')
    plt.bar(x + width, results_df[f'test_{metric}'], width, label='Test')
    
    plt.xlabel('Models')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xticks(x, results_df['model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()  

# Also create a single comprehensive plot showing F1 scores across all evaluation methods
plt.figure(figsize=(15, 8))
x = np.arange(len(results['model']))
width = 0.25

plt.bar(x - width, results_df['cv_f1'], width, label='CV F1 Score')
plt.bar(x, results_df['val_f1'], width, label='Validation F1 Score')
plt.bar(x + width, results_df['test_f1'], width, label='Test F1 Score')

# Add a horizontal line showing the average CV F1 score
avg_cv_f1 = np.mean(results_df['cv_f1'])
plt.axhline(y=avg_cv_f1, color='r', linestyle='--', label=f'Avg CV F1: {avg_cv_f1:.4f}')

plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('Comprehensive F1 Score Comparison Across Evaluation Methods')
plt.xticks(x, results_df['model'], rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/comprehensive_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nModel comparison visualizations saved to results/ directory")
print("\nTraining and evaluation complete!")