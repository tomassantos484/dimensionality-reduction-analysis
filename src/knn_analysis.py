# K-Nearest Neighbors Analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import time

# Small Dataset Imports
from preprocessing import x_small_dataset, y_small_dataset, x_small_dataset_scaled

# Medium Dataset Imports
from preprocessing import X_train_medium_dataset, y_train_medium_dataset, X_test_medium_dataset, y_test_medium_dataset, medium_dataset_features, medium_dataset_activity_labels

# Large Dataset Imports
from preprocessing import x_large_dataset_train_scaled, y_large_dataset_train, x_large_dataset_test_scaled, y_large_dataset_test, large_dataset_train, large_dataset_test

# Metric Tracking Import
from metric_tracking import log_knn_result

## SMALL DATASET K-NEIGHBOR ANALYSIS ##

# Split the small dataset into training and testing sets (70% training, 30% testing) with stratification
x_small_dataset_train_scaled, x_small_dataset_test_scaled, y_small_dataset_train, y_small_dataset_test = train_test_split(
    x_small_dataset_scaled, y_small_dataset, test_size=0.3, random_state=4, stratify=y_small_dataset)

# RUNTIME TRACKING START
small_dataset_knn_start_time = time.perf_counter()

# Print the start time
print(f"Small Dataset K-NN Start Time: {small_dataset_knn_start_time}")
print("="*100)

# kNN Analysis
kNN_small_dataset = KNeighborsClassifier(n_neighbors=5)
kNN_small_dataset.fit(x_small_dataset_train_scaled, y_small_dataset_train)

# Predict the testing set
y_small_dataset_pred = kNN_small_dataset.predict(x_small_dataset_test_scaled)
print("="*100)

# RUNTIME TRACKING END
small_dataset_knn_end_time = time.perf_counter()
print(f"Small Dataset K-NN End Time: {small_dataset_knn_end_time}")
print("="*100)
small_dataset_knn_total_time = small_dataset_knn_end_time - small_dataset_knn_start_time
print(f"Small Dataset K-NN Total Time: {small_dataset_knn_total_time} seconds")
print("="*100)

# Calculate the metrics
accuracy = accuracy_score(y_small_dataset_test, y_small_dataset_pred)
precision = precision_score(y_small_dataset_test, y_small_dataset_pred)
recall = recall_score(y_small_dataset_test, y_small_dataset_pred)
f1 = f1_score(y_small_dataset_test, y_small_dataset_pred)
y_probabilities = kNN_small_dataset.predict_proba(x_small_dataset_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_small_dataset_test, y_probabilities)

# Print the metrics
print("Small Dataset K-NN Metrics:")
print("="*100)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
print("="*100)

# Store metrics in csv
log_knn_result({
    'Dataset': 'Small',
    'Method': 'Baseline',
    'Dimensions': x_small_dataset_train_scaled.shape[1], # Dimensions of the training set
    'K-Nearest Neighbors Neighbors': 5,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC Score': roc_auc,
    'Time Taken (seconds)': small_dataset_knn_total_time,
})

## MEDIUM DATASET K-NEIGHBOR ANALYSIS ##

#RUNTIME TRACKING START
medium_dataset_knn_start_time = time.perf_counter()

# Print the start time
print(f"Medium Dataset K-NN Start Time: {medium_dataset_knn_start_time}")
print("="*100)

# kNN Analysis using already split training and testing sets | 6 neighbors for the 6 classes in the medium dataset
kNN_medium_dataset_train = KNeighborsClassifier(n_neighbors=6).fit(X_train_medium_dataset, y_train_medium_dataset)
y_medium_dataset_pred = kNN_medium_dataset_train.predict(X_test_medium_dataset)

# RUNTIME TRACKING END
medium_dataset_knn_end_time = time.perf_counter()
print(f"Medium Dataset K-NN End Time: {medium_dataset_knn_end_time}")
print("="*100)
medium_dataset_knn_total_time = medium_dataset_knn_end_time - medium_dataset_knn_start_time
print(f"Medium Dataset K-NN Total Time: {medium_dataset_knn_total_time} seconds")
print("="*100)

# Calculate the metrics
accuracy = accuracy_score(y_test_medium_dataset, y_medium_dataset_pred)
precision = precision_score(y_test_medium_dataset, y_medium_dataset_pred, average='macro', zero_division=0)
recall = recall_score(y_test_medium_dataset, y_medium_dataset_pred, average='macro')
f1 = f1_score(y_test_medium_dataset, y_medium_dataset_pred, average='macro')
y_probabilities = kNN_medium_dataset_train.predict_proba(X_test_medium_dataset)
roc_auc = roc_auc_score(y_test_medium_dataset, y_probabilities, multi_class='ovr', average='macro')

# Print the metrics
print("Medium Dataset K-NN Metrics:")
print("="*100)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
print("="*100)

# Store metrics in csv
log_knn_result({
    'Dataset': 'Medium',
    'Method': 'Baseline',
    'Dimensions': X_train_medium_dataset.shape[1], # Dimensions of the training set
    'K-Nearest Neighbors Neighbors': 5,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC Score': roc_auc,
    'Time Taken (seconds)': medium_dataset_knn_total_time,
})


## LARGE DATASET K-NEIGHBOR ANALYSIS ##

# RUNTIME TRACKING START
large_dataset_knn_start_time = time.perf_counter()

# Print the start time
print(f"Large Dataset K-NN Start Time: {large_dataset_knn_start_time}")
print("="*100)

# kNN Analysis using already split training and testing sets
kNN_large_dataset_train = KNeighborsClassifier(n_neighbors=5).fit(x_large_dataset_train_scaled, y_large_dataset_train)
y_large_dataset_pred = kNN_large_dataset_train.predict(x_large_dataset_test_scaled)

# RUNTIME TRACKING END
large_dataset_knn_end_time = time.perf_counter()
print(f"Large Dataset K-NN End Time: {large_dataset_knn_end_time}")
print("="*100)
large_dataset_knn_total_time = large_dataset_knn_end_time - large_dataset_knn_start_time
print(f"Large Dataset K-NN Total Time: {large_dataset_knn_total_time} seconds")
print("="*100)

# Calculate the metrics
accuracy = accuracy_score(y_large_dataset_test, y_large_dataset_pred)
precision = precision_score(y_large_dataset_test, y_large_dataset_pred, average='macro', zero_division=0)
recall = recall_score(y_large_dataset_test, y_large_dataset_pred, average='macro')
f1 = f1_score(y_large_dataset_test, y_large_dataset_pred, average='macro')
y_probabilities = kNN_large_dataset_train.predict_proba(x_large_dataset_test_scaled)
roc_auc = roc_auc_score(y_large_dataset_test, y_probabilities, multi_class='ovr', average='macro')

# Print the metrics
print("Large Dataset K-NN Metrics:")
print("="*100)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
print("="*100)

# Store metrics in csv
log_knn_result({
    'Dataset': 'Large',
    'Method': 'Baseline',
    'Dimensions': x_large_dataset_train_scaled.shape[1], # Dimensions of the training set
    'K-Nearest Neighbors Neighbors': 5,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC Score': roc_auc,
    'Time Taken (seconds)': large_dataset_knn_total_time,
})