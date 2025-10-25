import numpy as np

def sigmoid(x):
    # Standardization: zero mean, unit variance
    x_mean = np.mean(x)
    x_std = np.std(x)
    # Avoid division by zero by adding a small epsilon
    x_std = x_std if x_std != 0 else 1e-10
    x_normalized = (x - x_mean) / x_std
    # Clipping to prevent overflow
    x_clipped = np.clip(x_normalized, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

class LogisticRegression:
    def __init__(self, eta=0.001, epochs=1000):
        self.eta = eta
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.X_mean = None
        self.X_std = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Standardize X once at the start
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std == 0, 1e-10, self.X_std)  # Avoid division by zero
        X_normalized = (X - self.X_mean) / self.X_std
        
        for epoch in range(self.epochs):
            linear_pred = np.dot(X_normalized, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            
            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
            dw = (1 / n_samples) * np.dot(X_normalized.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            self.weights -= self.eta * dw
            self.bias -= self.eta * db
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Bias: {self.bias:.4f}")
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred