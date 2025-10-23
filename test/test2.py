import numpy as np
from KNN import KNN

# Create dummy data
X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([[2, 2], [5, 5]])

# Initialize and train
model = KNN(k=3)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("Predictions:", predictions)
