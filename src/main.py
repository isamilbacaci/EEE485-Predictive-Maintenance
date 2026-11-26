import pandas as pd
import numpy as np

df = pd.read_csv("data/ai4i2020.csv")

# Preprocessing
df = df.drop(['UDI', 'Product ID'], axis=1)
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1).values
y = df['Machine failure'].values

# Custom train-test split
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(len(X) * 0.2)

test_idx, train_idx = indices[:split_idx], indices[split_idx:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Z-score normalization (fitted only on train to prevent leakage)
mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)

X_train = (X_train - mu) / (sigma + 1e-8)
X_test = (X_test - mu) / (sigma + 1e-8)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# --- 1. Gaussian Discriminant Analysis (GDA) ---
class GDA:
    def fit(self, X, y):
        N, D = X.shape
        self.classes = np.unique(y)
        C = len(self.classes)

        self.mu = np.zeros((C, D))
        self.sigma2 = np.zeros((C, D))
        self.priors = np.zeros(C)

        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.mu[i, :] = Xc.mean(axis=0)
            self.sigma2[i, :] = Xc.var(axis=0) + 1e-9
            self.priors[i] = Xc.shape[0] / N
            
    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posts = []
        for i, _ in enumerate(self.classes):
            prior = np.log(self.priors[i])
            num = np.exp(- (x - self.mu[i])**2 / (2 * self.sigma2[i]))
            den = np.sqrt(2 * np.pi * self.sigma2[i])
            posts.append(prior + np.sum(np.log(num / den)))
        return self.classes[np.argmax(posts)]

gnb = GDA()
gnb.fit(X_train, y_train)
preds_gnb = gnb.predict(X_test)

# --- Helper: Evaluation Metrics ---
def get_metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate precision
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
        
    # Calculate recall
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
        
    # Calculate F1 score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return acc, tp, tn, fp, fn, precision, recall, f1

# GDA Results
acc_g, tp_g, tn_g, fp_g, fn_g, prec_g, rec_g, f1_g = get_metrics(y_test, preds_gnb)
print(f"GDA -> Acc: {acc_g:.4f} | TP: {tp_g}, TN: {tn_g}, FP: {fp_g}, FN: {fn_g} | Precision: {prec_g:.4f}, Recall: {rec_g:.4f}, F1: {f1_g:.4f}")

# --- 2. Weighted Logistic Regression (Ridge) ---
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, lambd=1.0, weights=None):
        self.lr = lr
        self.epochs = epochs
        self.lambd = lambd
        self.weights = weights 
        self.W = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        N, D = X.shape
        self.W = np.zeros(D)
        self.b = 0
        
        sw = np.ones(N)
        if self.weights:
            sw = np.array([self.weights[yi] for yi in y])

        for _ in range(self.epochs):
            y_hat = self.sigmoid(np.dot(X, self.W) + self.b)
            err = (y_hat - y) * sw

            dW = (1 / N) * np.dot(X.T, err) + (self.lambd / N) * self.W
            db = (1 / N) * np.sum(err)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        y_hat = self.sigmoid(np.dot(X, self.W) + self.b)
        return (y_hat > 0.5).astype(int)

lr_model = LogisticRegression(weights={0: 1.0, 1: 20.0})
lr_model.fit(X_train, y_train)
preds_lr = lr_model.predict(X_test)

# LR Results
acc_l, tp_l, tn_l, fp_l, fn_l, prec_l, rec_l, f1_l = get_metrics(y_test, preds_lr)
print(f"LR  -> Acc: {acc_l:.4f} | TP: {tp_l}, TN: {tn_l}, FP: {fp_l}, FN: {fn_l} | Precision: {prec_l:.4f}, Recall: {rec_l:.4f}, F1: {f1_l:.4f}")

# --- 3. Weighted MLP (1 Hidden Layer) ---
class MLP:
    def __init__(self, d_in, d_h=8, epochs=3000, lr=0.5, weights=None):
        self.d_in = d_in
        self.d_h = d_h
        self.epochs = epochs
        self.lr = lr
        self.weights = weights
        
        np.random.seed(42)
        self.W1 = np.random.randn(self.d_in, self.d_h) * 0.01
        self.b1 = np.zeros((1, self.d_h))
        self.W2 = np.random.randn(self.d_h, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        
    def fit(self, X, y):
        N = X.shape[0]
        Y = y.reshape(-1, 1)
        
        sw = np.ones((N, 1))
        if self.weights:
            sw = np.array([[self.weights[v[0]]] for v in Y])

        for _ in range(self.epochs):
            # Forward pass
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1) 
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)
            
            # Backward pass
            dZ2 = (A2 - Y) * sw
            dW2 = (1 / N) * np.dot(A1.T, dZ2)
            db2 = (1 / N) * np.sum(dZ2, axis=0, keepdims=True)
            
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * (A1 * (1 - A1))
            
            dW1 = (1 / N) * np.dot(X.T, dZ1)
            db1 = (1 / N) * np.sum(dZ1, axis=0, keepdims=True)
            
            # Weight update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        return (A2 > 0.5).astype(int).flatten()

mlp = MLP(d_in=X_train.shape[1], weights={0: 1.0, 1: 20.0})
mlp.fit(X_train, y_train)
preds_mlp = mlp.predict(X_test)

# MLP Results
acc_m, tp_m, tn_m, fp_m, fn_m, prec_m, rec_m, f1_m = get_metrics(y_test, preds_mlp)
print(f"MLP -> Acc: {acc_m:.4f} | TP: {tp_m}, TN: {tn_m}, FP: {fp_m}, FN: {fn_m} | Precision: {prec_m:.4f}, Recall: {rec_m:.4f}, F1: {f1_m:.4f}")

# --- 4. Feature Importance (Logistic Regression) ---
print("\n--- Feature Importance ---")
features = ['Type', 'Air Temp', 'Process Temp', 'Rotational Speed', 'Torque', 'Tool Wear']

# Get absolute values of weights
weights = np.abs(lr_model.W)

# Sort from highest to lowest
indices = np.argsort(weights)[::-1]

for i in indices:
    print(f"{features[i]}: {weights[i]:.4f}")