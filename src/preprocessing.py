# Data Loading and Preprocessing

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

## SMALL DATASET PREPROCESSING ##

# Load small dataset
small_dataset = pd.read_csv('data/small/high-dimensionality-kaggle/high_dimensionality.csv')

#Examine the dataset
print("Small Dataset:")
print(small_dataset.head()) # First 5 rows of the dataset
print("="*100)
print(small_dataset.info()) # Information about the dataset
print("="*100)
print(small_dataset.describe()) # Summary statistics of the dataset
print("="*100)
print(small_dataset.shape) # Dataset dimensions (230, 537)
print("="*100)
print(small_dataset.columns) # Column names
print("="*100)
print(small_dataset.index) # Index of the dataset
print("="*100)
print(small_dataset.dtypes) # Data types of the columns, 536 float64 and 1 int64
print("="*100)
print("Null values in the dataset: " + str(small_dataset.isnull().sum()))

# Rename columns to 'feature_1', 'feature_2', ..., 'feature_535' and keep 'Label' column intact
feature_columns = [col for col in small_dataset.columns if col != 'Label']
small_dataset = small_dataset.rename(
    columns={
        col: f"feature_{i + 1}"
        for i, col in enumerate(feature_columns)
    }
)
# Verify the changes
print(small_dataset.head())

# Split dataset into x and y
x_small_dataset = small_dataset.drop(columns=['Label'])
y_small_dataset = small_dataset['Label']

# Scale the dataset for kMeans
scaler = StandardScaler()
x_small_dataset_scaled = scaler.fit_transform(x_small_dataset)

# Convert the scaled dataset back to a pandas dataframe
x_small_dataset_scaled = pd.DataFrame(x_small_dataset_scaled, columns=x_small_dataset.columns)

print("X Small Dataset Scaled:")
print(x_small_dataset_scaled.head())
print("="*100)

## MEDIUM DATASET PREPROCESSING ##

# Note: The medium dataset is the UCI HAR Dataset. It is already preprocessed and ready to use.
X_train_medium_dataset = pd.read_csv('data/medium/uci-har-dataset/train/X_train.txt', header=None, sep='\s+')
y_train_medium_dataset = pd.read_csv('data/medium/uci-har-dataset/train/y_train.txt', header=None, sep='\s+').squeeze() # Converting to series as this is just a column of labels

X_test_medium_dataset = pd.read_csv('data/medium/uci-har-dataset/test/X_test.txt', header=None, sep='\s+')
y_test_medium_dataset = pd.read_csv('data/medium/uci-har-dataset/test/y_test.txt', header=None, sep='\s+').squeeze() # Converting to series as this is just a column of labels

medium_dataset_features = pd.read_csv('data/medium/uci-har-dataset/features.txt', header=None, sep='\s+')
medium_dataset_activity_labels = pd.read_csv('data/medium/uci-har-dataset/activity_labels.txt', header=None, sep='\s+')

print("X Train Medium Dataset:")
print(X_train_medium_dataset.head())
print("="*100)
print("Y Train Medium Dataset:")
print(y_train_medium_dataset.head())
print("="*100)
print("X Test Medium Dataset:")
print(X_test_medium_dataset.head())
print("="*100)
print("Y Test Medium Dataset:")
print(y_test_medium_dataset.head())
print("="*100)
print("Medium Dataset Features:")
print(medium_dataset_features.head())
print("="*100)
print("Medium Dataset Activity Labels:")
print(medium_dataset_activity_labels.head())

# LARGE DATASET PREPROCESSING ##

# Load large dataset
large_dataset_train = pd.read_csv('data/large/fashion-mnist/fashion-mnist_train.csv')
large_dataset_test = pd.read_csv('data/large/fashion-mnist/fashion-mnist_test.csv')

# Split large dataset into x and y
x_large_dataset_train = large_dataset_train.drop(columns=['label'])
y_large_dataset_train = large_dataset_train['label']

x_large_dataset_test = large_dataset_test.drop(columns=['label'])
y_large_dataset_test = large_dataset_test['label']

# Scaling dataset as recommended, dividing the pixel values by 255 for grayscale images
x_large_dataset_train_scaled = x_large_dataset_train.to_numpy() / 255.0
x_large_dataset_test_scaled = x_large_dataset_test.to_numpy() / 255.0

# Convert the scaled dataset back to a pandas dataframe
x_large_dataset_train_scaled = pd.DataFrame(x_large_dataset_train_scaled, columns=x_large_dataset_train.columns)
x_large_dataset_test_scaled = pd.DataFrame(x_large_dataset_test_scaled, columns=x_large_dataset_test.columns)

print("Large Dataset:")
print(large_dataset_train.head())
print("="*100)

print("Large Dataset Info:")
print(large_dataset_train.info())
print("="*100)

print("X Large Dataset Train Scaled:")
print(x_large_dataset_train_scaled.head())
print("="*100)
print("X Large Dataset Test Scaled:")
print(x_large_dataset_test_scaled.head())
print("="*100)

print("Y Large Dataset Train:")
print(y_large_dataset_train.head())
print("="*100)




