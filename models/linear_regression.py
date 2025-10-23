import numpy as np

class LinearRegression:
    
    def __init__(self, eta = 0.001, n_iters = 1000):
        """
        Initialize the LinearRegression model.
        
        Parameters:
        - eta (float): Learning rate for gradient descent (default: 0.001).
        - n_iters (int): Number of iterations for training (default: 1000).
        Initializes weights and bias as None, to be set during fit.
        """
        self.eta = eta
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Fit the model to the training data using gradient descent.
        
        Parameters:
        - X (numpy.ndarray): Training data features of shape (n_samples, n_features).
        - y (numpy.ndarray): Target values of shape (n_samples,).
        Initializes weights and bias, then iteratively updates them to minimize error.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            """
            Iterate n_iters times to optimize weights and bias.
            y_pred is the predicted value, dw and db are gradients for weights and bias.
            """
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.eta * dw
            self.bias -= self.eta * db
    
    def predict(self, X):
        """
        Predict target values for input data.
        
        Parameters:
        - X (numpy.ndarray): Input data features of shape (n_samples, n_features).
        Returns predicted values based on learned weights and bias.
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred