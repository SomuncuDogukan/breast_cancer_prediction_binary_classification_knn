"""
Project: Breast Cancer (Binary) Classification using KNN
Author: Dogukan Somuncu
Date: 2025
Description:
    A binary classification project using the Breast Cancer Wisconsin dataset. 
    Implements data preprocessing, training a KNN model, evaluating performance,
    and tuning the hyperparameter k (number of neighbors).
"""


# (1) Importing Required Libraries and Loading the Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt

# Load the breast cancer dataset
cancer = load_breast_cancer()
#print(cancer.DESCR)

# Create a DataFrame and add the target column
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target  # Binary target column: 0 (malignant), 1 (benign)


# (2) Data Preparation

# Define features (X) and target (y)
X = cancer.data
y = cancer.target

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# (3) Building and Training the KNN Model

# Initialize the KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model using training data
knn.fit(X_train, y_train)


# (4) Model Testing and Evaluation

# Make predictions on the test set
y_prediction = knn.predict(X_test)

# Calculate and print accuracy score
accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy:", accuracy)

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_prediction)
print("Confusion Matrix:")
print(conf_matrix)


# (5) Hyperparameter Tuning: Testing Different K Values
"""
    KNN Hyperparameter: K (number of neighbors)
    Try values from 1 to 20
    Plot accuracy for each K
"""

accuracy_values = []
k_values = []

# Test different values of K from 1 to 20
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

# Plot accuracy vs. K values
plt.figure()
plt.plot(k_values, accuracy_values, marker="o", linestyle="-")
plt.title("Accuracy vs K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
