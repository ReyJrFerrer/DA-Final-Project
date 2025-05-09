import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import os

# Create directory for ensemble results
os.makedirs('ensemble_results', exist_ok=True)

print("Loading preprocessed data...")
# Load validation and test data
X_val = np.load('models/X_val_scaled.npy')
y_val = np.load('models/y_val.npy')
X_test = np.load('models/X_test_scaled.npy')
y_test = np.load('models/y_test.npy')

# Load the saved models
print("Loading trained models...")
model_files = {
    'naive_bayes': 'models/naive_bayes_model.pkl',
    'svm': 'models/svm_model.pkl',
    'random_forest': 'models/random_forest_model.pkl',
    'decision_tree': 'models/decision_tree_model.pkl',
    'knn': 'models/knn_model.pkl',
    'xgboost': 'models/xgboost_model.pkl'
}

models = {}
for name, file_path in model_files.items():
    try:
        with open(file_path, 'rb') as f:
            models[name] = pickle.load(f)
            print(f"Loaded {name} model")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping {name} model.")

if len(models) < 2:
    print("Error: At least 2 models are required for ensembling.")
    exit(1)

# Define function to evaluate ensemble model
def evaluate_ensemble(ensemble, X, y, dataset_name):
    """Evaluate ensemble model and return metrics"""
    y_pred = ensemble.predict(X)
    y_pred_proba = ensemble.predict_proba(X)[:, 1]
    
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
    plt.title(f'Confusion Matrix - Ensemble ({dataset_name})')
    plt.savefig(f'ensemble_results/ensemble_{dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create and save ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Ensemble ({dataset_name})')
    plt.legend()
    plt.savefig(f'ensemble_results/ensemble_{dataset_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# Create ensemble models
print("\nCreating ensemble models...")

# 1. Create Voting Classifier with hard voting (majority vote)
estimators = [(name, model) for name, model in models.items()]
hard_voting_ensemble = VotingClassifier(estimators=estimators, voting='hard')
hard_voting_ensemble.fit(X_val, y_val)  # Fit on validation data to avoid overfitting

# 2. Create Voting Classifier with soft voting (probability-based)
soft_voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')
soft_voting_ensemble.fit(X_val, y_val)  # Fit on validation data to avoid overfitting

# Save ensemble models
with open('models/hard_voting_ensemble.pkl', 'wb') as f:
    pickle.dump(hard_voting_ensemble, f)
print("Hard voting ensemble saved to models/hard_voting_ensemble.pkl")

with open('models/soft_voting_ensemble.pkl', 'wb') as f:
    pickle.dump(soft_voting_ensemble, f)
print("Soft voting ensemble saved to models/soft_voting_ensemble.pkl")

# Evaluate ensemble models
print("\nEvaluating hard voting ensemble...")
hard_val_metrics = evaluate_ensemble(hard_voting_ensemble, X_val, y_val, 'validation_hard')
hard_test_metrics = evaluate_ensemble(hard_voting_ensemble, X_test, y_test, 'test_hard')

print("\nEvaluating soft voting ensemble...")
soft_val_metrics = evaluate_ensemble(soft_voting_ensemble, X_val, y_val, 'validation_soft')
soft_test_metrics = evaluate_ensemble(soft_voting_ensemble, X_test, y_test, 'test_soft')

# Create results DataFrame for comparison
# Load previous model results
previous_results = pd.read_csv('results/model_comparison.csv')

# Create ensemble results
ensemble_results = {
    'model': ['Hard Voting Ensemble', 'Soft Voting Ensemble'],
    'val_accuracy': [hard_val_metrics['accuracy'], soft_val_metrics['accuracy']],
    'val_precision': [hard_val_metrics['precision'], soft_val_metrics['precision']],
    'val_recall': [hard_val_metrics['recall'], soft_val_metrics['recall']],
    'val_f1': [hard_val_metrics['f1'], soft_val_metrics['f1']],
    'val_auc': [hard_val_metrics['auc'], soft_val_metrics['auc']],
    'test_accuracy': [hard_test_metrics['accuracy'], soft_test_metrics['accuracy']],
    'test_precision': [hard_test_metrics['precision'], soft_test_metrics['precision']],
    'test_recall': [hard_test_metrics['recall'], soft_test_metrics['recall']],
    'test_f1': [hard_test_metrics['f1'], soft_test_metrics['f1']],
    'test_auc': [hard_test_metrics['auc'], soft_test_metrics['auc']]
}
"""To fix"""
# Add train_time (just a placeholder as we didn't measure it for ensembles)
ensemble_results['train_time'] = [0, 0]  # We're not measuring training time for ensembles

# Combine previous results with ensemble results
ensemble_df = pd.DataFrame(ensemble_results)
combined_results = pd.concat([previous_results, ensemble_df])

# Save combined results
combined_results.to_csv('ensemble_results/all_models_comparison.csv', index=False)
print("\nCombined results saved to ensemble_results/all_models_comparison.csv")

# Visualize combined results
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
for metric in metrics:
    plt.figure(figsize=(14, 7))
    x = np.arange(len(combined_results['model']))
    width = 0.35
    
    plt.bar(x - width/2, combined_results[f'val_{metric}'], width, label='Validation')
    plt.bar(x + width/2, combined_results[f'test_{metric}'], width, label='Test')
    
    plt.xlabel('Models')
    plt.ylabel(metric.capitalize())
    plt.title(f'All Models Comparison - {metric.capitalize()}')
    plt.xticks(x, combined_results['model'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'ensemble_results/all_models_comparison_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nEnsemble model evaluation complete!")
print("\nResults summary:")
print(f"Hard Voting Ensemble - Test: Accuracy={hard_test_metrics['accuracy']:.4f}, "
      f"Precision={hard_test_metrics['precision']:.4f}, Recall={hard_test_metrics['recall']:.4f}, "
      f"F1={hard_test_metrics['f1']:.4f}, AUC={hard_test_metrics['auc']:.4f}")
print(f"Soft Voting Ensemble - Test: Accuracy={soft_test_metrics['accuracy']:.4f}, "
      f"Precision={soft_test_metrics['precision']:.4f}, Recall={soft_test_metrics['recall']:.4f}, "
      f"F1={soft_test_metrics['f1']:.4f}, AUC={soft_test_metrics['auc']:.4f}")