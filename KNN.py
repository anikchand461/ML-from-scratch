import numpy as np
from collections import Counter

def eucludian_diatance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2) ** 2))  # Compute Euclidean distance: subtract vectors, square differences, sum them, and take square root
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):  # Method to 'train' the model by storing training data (KNN is a lazy learning algorithm)
        self.X_train = X  # Store the training feature matrix X as an instance variable
        self.y_train = y  # Store the training labels y as an instance variable

    def predict(self, X):  # Method to predict class labels for test data X
        predictions = [self._predict(x) for x in X]  # Use list comprehension to predict label for each test sample by calling _predict
        return predictions  # Return the list of predicted labels
    
    def _predict(self, x):  # Helper method to predict the class label for a single test sample x
        # compute distances 
        distances = [eucludian_diatance(x, x_train) for x_train in self.X_train]  # Calculate Euclidean distance from x to each training sample
        
        # get the closest k (minimum k distances)
        k_indices = np.argsort(distances)[:self.k]  # Sort distances, get indices of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]  # Get the labels of the k nearest neighbors
        
        # majority vote
        most_common = Counter(k_nearest_labels).most_common()  # Count occurrences of each label and get the most common one
        # counter returns dictionary with keys and values (frequency) and then when applied the most_common() fn then it will convert to list of tuples - [(key1, val1), (key2, val2)...]
        
        return most_common[0][0].tolist()  # Return the labels with the highest counts (majority votes) - if tie then return the 0th elements