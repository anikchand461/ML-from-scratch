# main.py (in test/ folder)
import sys
import os
# Add parent directory to path for imports (fallback if relative import fails)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from KNN import KNN  # Absolute import (works with sys.path)
# from ..KNN import KNN  # Alternative: relative import (requires __init__.py and python -m)

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

# Custom colormap for visualization
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Initialize and train KNN (using features 2 and 3: petal length/width for simplicity)
knn = KNN(k=3)
knn.fit(X_train[:, [2, 3]], y_train)  # Use only 2D features for easy plotting

# Predict on test set
predictions = knn.predict(X_test[:, [2, 3]])

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"KNN Accuracy: {accuracy:.4f}")

# Optional: Visualize decision boundary and predictions
def plot_decision_boundary():
    # Create mesh for decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict on mesh
    mesh_predictions = np.array(knn.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, mesh_predictions, alpha=0.3, cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors='k', s=30)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap=cmap, marker='^', edgecolors='k', s=30, alpha=0.7)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(f'KNN Decision Boundary (k=3, Accuracy: {accuracy:.4f})')
    plt.show()

# Run visualization (comment out if no display)
# plot_decision_boundary()