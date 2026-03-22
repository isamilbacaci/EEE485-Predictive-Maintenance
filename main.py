import pandas as pd
import numpy as np

print("Loading the dataset...")
df = pd.read_csv("ai4i2020.csv")

# 1. Drop useless columns (Identifiers that hold no predictive value)
df = df.drop(['UDI', 'Product ID'], axis=1)

# 2. Encode categorical data to numerical values
# 'Type' column has 'L' (Low), 'M' (Medium), 'H' (High) quality variants.
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# 3. Separate Features (X) and Target (y)
# 'Machine failure' is our binary target.
# We must drop specific failure type columns (TWF, HDF, PWF, OSF, RNF) to prevent data leakage.
X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1).values
y = df['Machine failure'].values

# 4. Train-Test Split from scratch (80% Train, 20% Test)
np.random.seed(42) # Set seed for reproducibility
indices = np.random.permutation(len(X))
test_size = int(len(X) * 0.2)

test_indices = indices[:test_size]
train_indices = indices[test_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# 5. Standardization from scratch (Z-score normalization)
# Formula: z = (x - mean) / std
# CRITICAL: Calculate mean and std ONLY on the training set to prevent data leakage from the test set.
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Add a tiny epsilon (1e-8) to standard deviation to prevent division by zero
X_train_scaled = (X_train - mean) / (std + 1e-8)
X_test_scaled = (X_test - mean) / (std + 1e-8)

print("\n--- Preprocessing Completed ---")
print(f"Training features shape: {X_train_scaled.shape}")
print(f"Testing features shape: {X_test_scaled.shape}")
print(f"Number of failures in training set: {np.sum(y_train)} out of {len(y_train)}")