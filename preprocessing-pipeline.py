import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('email_phishing_data.csv')

# Convert label to is_phishing for better readability
df['is_phishing'] = df['label'].astype(bool)
df = df.drop('label', axis=1)

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Split features and target
X = df.drop('is_phishing', axis=1)
y = df['is_phishing']

# 1. Perform train/validation/test split (70% / 15% / 15%)
# First split into train and temp (temp will be split into validation and test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Then split temp into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Check class distribution in splits
print("\nClass distribution:")
print(f"Training set: {np.bincount(y_train.astype(int))}")
print(f"Validation set: {np.bincount(y_val.astype(int))}")
print(f"Test set: {np.bincount(y_test.astype(int))}")
print(f"Original imbalance ratio: {np.bincount(y.astype(int))[0] / np.bincount(y.astype(int))[1]:.2f}")

# 2. Standardize numerical features
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to models/scaler.pkl")

# 3. Handle class imbalance using SMOTE (only on training data)
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - Training samples: {X_train_resampled.shape[0]}")
print(f"After SMOTE - Class distribution: {np.bincount(y_train_resampled.astype(int))}")

# Save preprocessed datasets for model training
print("\nSaving preprocessed datasets...")
np.save('models/X_train_resampled.npy', X_train_resampled)
np.save('models/y_train_resampled.npy', y_train_resampled)
np.save('models/X_val_scaled.npy', X_val_scaled)
np.save('models/y_val.npy', y_val.values)
np.save('models/X_test_scaled.npy', X_test_scaled)
np.save('models/y_test.npy', y_test.values)

print("Preprocessed datasets saved to models/ directory")

# Visualize the resampled data
print("\nVisualizing feature distributions after preprocessing...")

# Convert numpy arrays back to dataframes for easier plotting
X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
X_train_resampled_df['is_phishing'] = y_train_resampled

# Plot distributions of top features after SMOTE
top_features = ['num_words', 'num_unique_words', 'num_stopwords', 'num_links', 'num_spelling_errors']  # Using actual feature names from dataset

for feature in top_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=X_train_resampled_df, x=feature, hue='is_phishing', 
                 kde=True, element='step', palette=['blue', 'red'], 
                 hue_order=[False, True], fill=False)
    plt.title(f'Distribution of {feature} after SMOTE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/after_smote_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Feature distribution plots saved to plots/ directory")
print("\nPreprocessing complete! Data is ready for model training.")