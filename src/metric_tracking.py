# Helper file for tracking metrics for this project

# Import libraries
import csv
import pathlib
from typing import Iterable


# K-Means Columns
kmeans_columns = [
    'Dataset',
    'Method',
    'Dimensions',
    'K-Means Clusters',
    'n_init',
    'random_state',
    'Adjusted Rand Index (ARI)',
    'Normalized Mutual Information (NMI)',
    'Silhouette Score (SS)',
    'SS Sample Size',
    'SS Random State',
    'Time Taken (seconds)',
]

#K-Nearest Neighbors Columns
knn_columns = [
    'Dataset',
    'Method',
    'Dimensions',
    'K-Nearest Neighbors Neighbors',
    'Accuracy',
    'Precision',
    'Recall',
    'F1 Score',
    'ROC AUC Score',
    'Time Taken (seconds)',
]

# Principal Component Analysis (PCA) Columns
pca_columns = [
    'Dataset',
    'Method',
    'Dimensions',
    'PCA Components',
    'Explained Variance Score',
    'Adjusted Rand Index (ARI)',
    'Normalized Mutual Information (NMI)',
    'Silhouette Score (SS)',
]

# Global Variables for paths
KMEANS_METRICS_PATH = pathlib.Path('results/metrics/kmeans_metrics.csv')
KNN_METRICS_PATH = pathlib.Path('results/metrics/knn_metrics.csv')

def log_result(path: pathlib.Path | str, row: dict, columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged result to {path} successfully!")


def log_kmeans_result(row: dict) -> None:
    log_result(KMEANS_METRICS_PATH, row, kmeans_columns)


def log_knn_result(row: dict) -> None:
    log_result(KNN_METRICS_PATH, row, knn_columns)


def reset_kmeans_metrics() -> None:
    KMEANS_METRICS_PATH.unlink(missing_ok=True)


def reset_knn_metrics() -> None:
    KNN_METRICS_PATH.unlink(missing_ok=True)
