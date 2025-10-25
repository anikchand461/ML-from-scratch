import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

lr = LogisticRegression(epochs=100000, eta=0.1)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(y_pred)

acc = accuracy_score(y_pred, y_test)

print(acc)