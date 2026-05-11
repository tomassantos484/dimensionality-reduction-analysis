# K-Means Analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Small Dataset Imports
from preprocessing import x_small_dataset, y_small_dataset, x_small_dataset_scaled

# Medium Dataset Imports
from preprocessing import X_train_medium_dataset, y_train_medium_dataset, X_test_medium_dataset, y_test_medium_dataset, medium_dataset_features, medium_dataset_activity_labels

# Large Dataset Imports
from preprocessing import x_large_dataset_train_scaled, y_large_dataset_train, x_large_dataset_test_scaled, y_large_dataset_test, large_dataset_train, large_dataset_test

# Metric Tracking Imports
from metric_tracking import log_kmeans_result

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans

## SMALL DATASET K-MEANS ANALYSIS ##

# Plot the dataset first, first 2 features vs last 2 features
plt.figure(figsize=(10, 10))
plt.scatter(
    x_small_dataset.iloc[:, 0],
    x_small_dataset.iloc[:, 1],
    c=y_small_dataset,
    cmap='viridis'
)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Small Dataset First 2 Features')

# Save it to a file in results
#plt.savefig('results/graphs/pre-DR/small_dataset_first_2_features_plot.png')

# Show the plot
#plt.show()


plt.figure(figsize=(10, 10))
plt.scatter(
    x_small_dataset.iloc[:, -2],
    x_small_dataset.iloc[:, -1],
    c=y_small_dataset,
    cmap='viridis'
)
plt.xlabel('Feature 534')
plt.ylabel('Feature 535')
plt.title('Small Dataset Last 2 Features')

# Save it to a file in results
#plt.savefig('results/graphs/pre-DR/small_dataset_last_2_features_plot.png')

# Show the plot
#plt.show()

# RUNTIME TRACKING START
small_dataset_kmeans_start_time = time.perf_counter()

# Print the start time
print(f"Small Dataset K-Means Start Time: {small_dataset_kmeans_start_time}")
print("="*100)

# Apply K-Means clustering
kmeans_small_dataset = KMeans(n_clusters=2, n_init=10, random_state=42).fit(x_small_dataset_scaled)
labels_small_dataset = kmeans_small_dataset.labels_

# Gather metrics (ARI, NMI, Silhouette Score)
ari = adjusted_rand_score(y_small_dataset, labels_small_dataset)
nmi = normalized_mutual_info_score(y_small_dataset, labels_small_dataset)
silhouette = silhouette_score(x_small_dataset_scaled, labels_small_dataset)

# RUNTIME TRACKING END
small_dataset_kmeans_end_time = time.perf_counter()
print(f"Small Dataset K-Means End Time: {small_dataset_kmeans_end_time}")
print("="*100)
small_dataset_kmeans_total_time = small_dataset_kmeans_end_time - small_dataset_kmeans_start_time
print(f"Small Dataset K-Means Total Time: {small_dataset_kmeans_total_time} seconds")
print("="*100)

# Store metrics in csv
log_kmeans_result({
    'Dataset': 'Small',
    'Method': 'Baseline',
    'Dimensions': x_small_dataset_scaled.shape[1], # Dimensions of the scaled dataset
    'K-Means Clusters': 2,
    'n_init': 10,
    'random_state': 42,
    'Adjusted Rand Index (ARI)': ari,
    'Normalized Mutual Information (NMI)': nmi,
    'Silhouette Score (SS)': silhouette,
    'SS Sample Size': None,
    'SS Random State': None,
    'Time Taken (seconds)': small_dataset_kmeans_total_time,
})


# Plot the small dataset with the K-Means clusters, first 2 features
plt.figure(figsize=(10, 10))
plt.scatter(x_small_dataset_scaled.iloc[:, 0], 
x_small_dataset_scaled.iloc[:, 1], 
c=labels_small_dataset, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Small Dataset First 2 Features with K-Means Clusters')

# Save it to a file in results
#plt.savefig('results/graphs/pre-DR/small_dataset_first_2_features_with_kmeans_plot.png')

# Show the plot
#plt.show()

# Plot the small dataset with the K-Means clusters, last 2 features
plt.figure(figsize=(10, 10))
plt.scatter(x_small_dataset_scaled.iloc[:, -2], x_small_dataset_scaled.iloc[:, -1], c=labels_small_dataset, cmap='viridis')
plt.xlabel('Feature 534')
plt.ylabel('Feature 535')
plt.title('Small Dataset Last 2 Features with K-Means Clusters')

# Save it to a file in results
#plt.savefig('results/graphs/pre-DR/small_dataset_last_2_features_with_kmeans_plot.png')

# Show the plot
#plt.show()

## MEDIUM DATASET K-MEANS ANALYSIS ##

# Plot medium dataset
plt.figure(figsize=(10, 10))
plt.scatter(X_train_medium_dataset.iloc[:, 0], X_train_medium_dataset.iloc[:, 1], c=y_train_medium_dataset, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Medium Dataset First 2 Features')

# Save it to a file in results
#plt.savefig('results/graphs/pre-DR/medium_dataset_first_2_features_plot.png')

# Show the plot
#plt.show()

# RUNTIME TRACKING START
medium_dataset_kmeans_start_time = time.perf_counter()

# Print the start time
print(f"Medium Dataset K-Means Start Time: {medium_dataset_kmeans_start_time}")
print("="*100)

# K-means clustering on medium dataset
kmeans_medium_dataset = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X_train_medium_dataset) # 6 clusters as there are 6 activities tracked in the dataset
labels_medium_dataset = kmeans_medium_dataset.labels_

print("Sorted Labels:")
print(sorted(set(labels_medium_dataset)))
print("="*100)


# Gather metrics (ARI, NMI, Silhouette Score)
ari = adjusted_rand_score(y_train_medium_dataset, labels_medium_dataset)
nmi = normalized_mutual_info_score(y_train_medium_dataset, labels_medium_dataset)
silhouette = silhouette_score(X_train_medium_dataset, labels_medium_dataset)

# RUNTIME TRACKING END
medium_dataset_kmeans_end_time = time.perf_counter()
print(f"Medium Dataset K-Means End Time: {medium_dataset_kmeans_end_time}")
print("="*100)

medium_dataset_kmeans_total_time = medium_dataset_kmeans_end_time - medium_dataset_kmeans_start_time
print(f"Medium Dataset K-Means Total Time: {medium_dataset_kmeans_total_time} seconds")
print("="*100)

# Store metrics in csv
log_kmeans_result({
    'Dataset': 'Medium',
    'Method': 'Baseline',
    'Dimensions': X_train_medium_dataset.shape[1], # Dimensions of the training set
    'K-Means Clusters': 6,
    'n_init': 10,
    'random_state': 42,
    'Adjusted Rand Index (ARI)': ari,
    'Normalized Mutual Information (NMI)': nmi,
    'Silhouette Score (SS)': silhouette,
    'SS Sample Size': None,
    'SS Random State': None,
    'Time Taken (seconds)': medium_dataset_kmeans_total_time,
})
# Plot the medium dataset with the K-Means clusters, first 2 features
plt.figure(figsize=(10, 10))
plt.scatter(X_train_medium_dataset.iloc[:, 0], X_train_medium_dataset.iloc[:, 1], c=labels_medium_dataset, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Medium Dataset First 2 Features with K-Means Clusters')

# Save it to a file in results
#plt.savefig('results/graphs/pre-DR/medium_dataset_first_2_features_with_kmeans_plot.png')

# Show the plot
#plt.show()

# LARGE DATASET K-MEANS ANALYSIS ##

# Plot a random set of 1000 pixels from the large dataset
rng = np.random.default_rng(42)
sample_index = rng.choice(x_large_dataset_train_scaled.index, size=1000, replace=False)

x_large_sample = x_large_dataset_train_scaled.iloc[sample_index]
y_large_sample = y_large_dataset_train.iloc[sample_index]

large_sample_labels = y_large_dataset_train.iloc[sample_index]

# Plot the large sample of 1000 pixels
plt.figure(figsize=(10, 10))
plt.scatter(x_large_sample.iloc[:, 0], x_large_sample.iloc[:, 1], c=large_sample_labels, cmap='tab10', alpha=0.5)
plt.xlabel('Pixel 1')
plt.ylabel('Pixel 2')
plt.title('Fashion-MNIST pixel 1 v pixel 2 (1000 samples)')

# Save it to a file in results
#plt.savefig('results/graphs/pre-DR/fashion-mnist_pixel_1_v_pixel_2_1000_samples.png')

# Show the plot
#plt.show() # No useful information to be gained from this plot


# Plotting pixels in middle of 28 x 28 image (pixels 400 and 420)

plt.figure(figsize=(10, 10))
plt.scatter(x_large_sample.iloc[:, 400], x_large_sample.iloc[:, 420], c=large_sample_labels, cmap='tab10', alpha=0.5)
plt.xlabel('Pixel 400')
plt.ylabel('Pixel 420')
plt.title('Fashion-MNIST pixel 400 v pixel 420 (1000 samples)')

# Save it to a file in results
plt.savefig('results/graphs/pre-DR/fashion-mnist_pixel_400_v_pixel_420_1000_samples.png')

# Show the plot
plt.show() # Much more useful information to be gained from this plot!

# RUNTIME TRACKING START
large_dataset_kmeans_start_time = time.perf_counter()

# Print the start time
print(f"Large Dataset K-Means Start Time: {large_dataset_kmeans_start_time}")
print("="*100)

# Kmeans clustering on large dataset, 1000 samples
kmeans_large_dataset = KMeans(n_clusters=10, n_init=10, random_state=42).fit(x_large_dataset_train_scaled)

labels_large_dataset = kmeans_large_dataset.labels_

# Gathering metrics (ARI, NMI, and Silhouette Score with sample size of 1000)
ari = adjusted_rand_score(y_large_dataset_train, labels_large_dataset)
nmi = normalized_mutual_info_score(y_large_dataset_train, labels_large_dataset)
silhouette = silhouette_score(x_large_dataset_train_scaled, labels_large_dataset,
sample_size=1000, random_state=42)


# Print the metrics
print("Large Dataset K-Means Metrics:")
print("="*100)
print(f"Adjusted Rand Index: {ari}")
print(f"Normalized Mutual Information: {nmi}")
print(f"Silhouette Score: {silhouette}")

# RUNTIME TRACKING END
large_dataset_kmeans_end_time = time.perf_counter()
print(f"Large Dataset K-Means End Time: {large_dataset_kmeans_end_time}")
print("="*100)

large_dataset_kmeans_total_time = large_dataset_kmeans_end_time - large_dataset_kmeans_start_time
print(f"Large Dataset K-Means Total Time: {large_dataset_kmeans_total_time} seconds")
print("="*100)

# Store metrics in csv
log_kmeans_result({
    'Dataset': 'Large',
    'Method': 'Baseline',
    'Dimensions': x_large_dataset_train_scaled.shape[1], # Dimensions of the training set
    'K-Means Clusters': 10,
    'n_init': 10,
    'random_state': 42,
    'Adjusted Rand Index (ARI)': ari,
    'Normalized Mutual Information (NMI)': nmi,
    'Silhouette Score (SS)': silhouette,
    'SS Sample Size': 1000,
    'SS Random State': 42,
    'Time Taken (seconds)': large_dataset_kmeans_total_time,
})

# Kmeans clustering on large dataset, 2000 samples
kmeans_large_dataset = KMeans(n_clusters=10, n_init=10, random_state=42).fit(x_large_dataset_train_scaled)

labels_large_dataset = kmeans_large_dataset.labels_

# Gathering metrics (ARI, NMI, and Silhouette Score with sample size of 1000)
ari = adjusted_rand_score(y_large_dataset_train, labels_large_dataset)
nmi = normalized_mutual_info_score(y_large_dataset_train, labels_large_dataset)
silhouette = silhouette_score(x_large_dataset_train_scaled, labels_large_dataset,
sample_size=2000, random_state=42)


# Print the metrics
print("Large Dataset K-Means Metrics:")
print("="*100)
print(f"Adjusted Rand Index: {ari}")
print(f"Normalized Mutual Information: {nmi}")
print(f"Silhouette Score: {silhouette}")

# RUNTIME TRACKING END
large_dataset_kmeans_end_time = time.perf_counter()
print(f"Large Dataset K-Means End Time: {large_dataset_kmeans_end_time}")
print("="*100)

large_dataset_kmeans_total_time = large_dataset_kmeans_end_time - large_dataset_kmeans_start_time
print(f"Large Dataset K-Means Total Time: {large_dataset_kmeans_total_time} seconds")
print("="*100)

# Store metrics in csv
log_kmeans_result({
    'Dataset': 'Large',
    'Method': 'Baseline',
    'Dimensions': x_large_dataset_train_scaled.shape[1], # Dimensions of the training set
    'K-Means Clusters': 10,
    'n_init': 10,
    'random_state': 42,
    'Adjusted Rand Index (ARI)': ari,
    'Normalized Mutual Information (NMI)': nmi,
    'Silhouette Score (SS)': silhouette,
    'SS Sample Size': 2000,
    'SS Random State': 42,
    'Time Taken (seconds)': large_dataset_kmeans_total_time,
})


# Kmeans clustering on large dataset, entire dataset
# Kmeans clustering on large dataset, 1000 samples
kmeans_large_dataset = KMeans(n_clusters=10, n_init=10, random_state=42).fit(x_large_dataset_train_scaled)

labels_large_dataset = kmeans_large_dataset.labels_

# Gathering metrics (ARI, NMI, and Silhouette Score with sample size of 1000)
ari = adjusted_rand_score(y_large_dataset_train, labels_large_dataset)
nmi = normalized_mutual_info_score(y_large_dataset_train, labels_large_dataset)
silhouette = silhouette_score(x_large_dataset_train_scaled, labels_large_dataset)


# Print the metrics
print("Large Dataset K-Means Metrics:")
print("="*100)
print(f"Adjusted Rand Index: {ari}")
print(f"Normalized Mutual Information: {nmi}")
print(f"Silhouette Score: {silhouette}")

# RUNTIME TRACKING END
large_dataset_kmeans_end_time = time.perf_counter()
print(f"Large Dataset K-Means End Time: {large_dataset_kmeans_end_time}")
print("="*100)

large_dataset_kmeans_total_time = large_dataset_kmeans_end_time - large_dataset_kmeans_start_time
print(f"Large Dataset K-Means Total Time: {large_dataset_kmeans_total_time} seconds")
print("="*100)

# Store metrics in csv
log_kmeans_result({
    'Dataset': 'Large',
    'Method': 'Baseline',
    'Dimensions': x_large_dataset_train_scaled.shape[1], # Dimensions of the training set
    'K-Means Clusters': 10,
    'n_init': 10,
    'random_state': 42,
    'Adjusted Rand Index (ARI)': ari,
    'Normalized Mutual Information (NMI)': nmi,
    'Silhouette Score (SS)': silhouette,
    'SS Sample Size': None,
    'SS Random State': None,
    'Time Taken (seconds)': large_dataset_kmeans_total_time,
})
