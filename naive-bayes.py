import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import os

# Create directories for results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/naive_bayes', exist_ok=True)

# Define model name
MODEL_NAME = 'Naive Bayes'

# Load preprocessed data
print(f"\n{'='*50}")
print(f"Training and evaluating {MODEL_NAME} model")
print(f"{'='*50}")

print("Loading preprocessed data...")
X_train = np.load('models/X_train_resampled.npy')
y_train = np.load('models/y_train_resampled.npy')
X_val = np.load('models/X_val_scaled.npy')
y_val = np.load('models/y_val.npy')
X_test = np.load('models/X_test_scaled.npy')
y_test = np.load('models/y_test.npy')

# Define model
model = GaussianNB()

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
    plt.savefig(f'results/naive_bayes/{model_name}_{dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'results/naive_bayes/{model_name}_{dataset_name}_roc_curve.png', dpi=300, bbox_inches='tight')
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
    'model': [MODEL_NAME],
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

# Setup cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on training data
start_time = time.time()
cv_metrics = cross_validate_model(model, X_train, y_train, skf, MODEL_NAME)
cv_time = time.time() - start_time
print(f"Cross-validation completed in {cv_time:.2f} seconds")

# Train final model on the whole training set
print(f"Training final {MODEL_NAME} model on full training set...")
model_final = pickle.loads(pickle.dumps(model))  # Create a deep copy
start_time = time.time()
model_final.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds")

# Save the final model
model_filename = f'models/{MODEL_NAME.lower().replace(" ", "_")}_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_final, f)
print(f"Model saved to {model_filename}")

# Evaluate on validation set
print(f"Evaluating {MODEL_NAME} on validation set...")
val_metrics = evaluate_model(model_final, X_val, y_val, MODEL_NAME, 'validation')

# Evaluate on test set
print(f"Evaluating {MODEL_NAME} on test set...")
test_metrics = evaluate_model(model_final, X_test, y_test, MODEL_NAME, 'test')

# Store results
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

print(f"{MODEL_NAME} evaluation completed.")
print(f"CV: Accuracy={cv_metrics['accuracy']:.4f}, Precision={cv_metrics['precision']:.4f}, "
      f"Recall={cv_metrics['recall']:.4f}, F1={cv_metrics['f1']:.4f}, AUC={cv_metrics['auc']:.4f}")
print(f"Validation: Accuracy={val_metrics['accuracy']:.4f}, Precision={val_metrics['precision']:.4f}, "
      f"Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")
print(f"Test: Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, "
      f"Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")

# Create results DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f'results/naive_bayes/naive_bayes_results.csv', index=False)
print(f"\nResults saved to results/naive_bayes/naive_bayes_results.csv")

print("\nTraining and evaluation complete!")