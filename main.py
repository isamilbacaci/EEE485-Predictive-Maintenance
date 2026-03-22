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

# =====================================================================
# MODEL 1: GAUSSIAN NAIVE BAYES (Generative Approach) FROM SCRATCH
# =====================================================================
print("\n--- Training Gaussian Naive Bayes ---")

class GaussianNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Matrisleri sıfırlarla başlatıyoruz
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c] # Sadece o sınıfa (Arızalı veya Sağlam) ait verileri filtrele
            self.mean[idx, :] = X_c.mean(axis=0)
            # Sıfıra bölünme (divide by zero) hatasını önlemek için varyansa çok küçük bir epsilon ekliyoruz
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(n_samples)
            
    def predict(self, X):
        # Test setindeki her bir sensör verisi (satır) için tahmin yap
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # Her sınıf (0: Sağlam, 1: Arızalı) için olasılık hesabı
        # Çarpım işlemi çok küçük sayılar üreteceği için "Log Probability" kullanıyoruz (Underflow'u önler)
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            # Gauss Olasılık Yoğunluk Fonksiyonu (PDF) logaritması
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # En yüksek olasılığa sahip sınıfı döndür (Argmax)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Modeli Eğit (Train)
gnb = GaussianNaiveBayes()
gnb.fit(X_train_scaled, y_train)

# Test seti üzerinde tahmin yap (Predict)
predictions = gnb.predict(X_test_scaled)

# =====================================================================
# EVALUATION METRICS FROM SCRATCH
# =====================================================================
def evaluate_performance(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # Karmaşıklık Matrisi (Confusion Matrix) hesaplama
    TP = np.sum((y_true == 1) & (y_pred == 1)) # True Positive: Doğru bilinen arızalar
    TN = np.sum((y_true == 0) & (y_pred == 0)) # True Negative: Doğru bilinen sağlamlar
    FP = np.sum((y_true == 0) & (y_pred == 1)) # False Positive: Yanlış alarm (Sağlam makineye arızalı demek)
    FN = np.sum((y_true == 1) & (y_pred == 0)) # False Negative: Gözden kaçan arızalar! (En tehlikelisi)
    
    return accuracy, TP, TN, FP, FN

acc, TP, TN, FP, FN = evaluate_performance(y_test, predictions)

print(f"Gaussian Naive Bayes Accuracy: {acc * 100:.2f}%")
print("\n--- Confusion Matrix ---")
print(f"True Positives (Correctly Predicted Failures): {TP}")
print(f"True Negatives (Correctly Predicted Normals): {TN}")
print(f"False Positives (False Alarms): {FP}")
print(f"False Negatives (Missed Failures): {FN}")

# =====================================================================
# MODEL 2: L2-REGULARIZED LOGISTIC REGRESSION FROM SCRATCH
# =====================================================================
print("\n--- Training Logistic Regression (L2 Regularized) ---")

class LogisticRegressionL2:
    def __init__(self, learning_rate=0.1, epochs=2000, lambda_param=1.0):
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param # L2 (Ridge) Regularization parametresi
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # np.exp'in çok büyük/küçük sayılarda patlamasını (overflow) önlemek için z'yi sınırlandırıyoruz
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Ağırlıkları ve bias'ı sıfırlarla başlat (Zero Initialization)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Döngüsü (İteratif Öğrenme)
        for _ in range(self.epochs):
            # İleri Besleme (Forward Pass)
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Türevleri (Gradients) Hesapla
            # L2 penalty = (lambda / n_samples) * weights. (Bias parametresi penalize edilmez)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.lambda_param / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Ağırlıkları Güncelle (Update Weights)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Olasılık %50'den büyükse 1 (Arızalı), küçükse 0 (Sağlam) sınıfına ata
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Modeli Eğit (Train)
log_reg = LogisticRegressionL2(learning_rate=0.1, epochs=2000, lambda_param=1.0)
log_reg.fit(X_train_scaled, y_train)

# Test seti üzerinde tahmin yap (Predict)
log_preds = log_reg.predict(X_test_scaled)

# Performansı Değerlendir
acc_lr, TP_lr, TN_lr, FP_lr, FN_lr = evaluate_performance(y_test, log_preds)

print(f"Logistic Regression Accuracy: {acc_lr * 100:.2f}%")
print("\n--- Confusion Matrix ---")
print(f"True Positives (Correctly Predicted Failures): {TP_lr}")
print(f"True Negatives (Correctly Predicted Normals): {TN_lr}")
print(f"False Positives (False Alarms): {FP_lr}")
print(f"False Negatives (Missed Failures): {FN_lr}")