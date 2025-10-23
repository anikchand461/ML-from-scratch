import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
plt.show()

lr = LinearRegression(eta=0.01, n_iters=50000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse, r2)

y_pred_line = lr.predict(X)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, c='b', linewidth=2, label='Prediction')
plt.show()