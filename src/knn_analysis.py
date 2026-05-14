# K-Nearest Neighbors Analysis (baseline / pre-DR).
#
# Runs kNN on each dataset at full feature dimensionality and writes one row
# per dataset to results/metrics/practice-runs/knn_metrics.csv. The
# multi-seed sweep in src/experiment_runner.py is the canonical entry point;
# this file is kept as a focused, single-seed reference.

import time

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing import (
    x_small_dataset_scaled, y_small_dataset,
    X_train_medium_dataset, y_train_medium_dataset,
    X_test_medium_dataset, y_test_medium_dataset,
    x_large_dataset_train_scaled, y_large_dataset_train,
    x_large_dataset_test_scaled, y_large_dataset_test,
)
from src.metric_tracking import log_knn_result

N_NEIGHBORS = 5
RANDOM_STATE = 42
SEP = "=" * 100


def _print_metrics(name, accuracy, precision, recall, f1, roc_auc, elapsed):
    print(f"{name} K-NN Total Time: {elapsed:.4f}s")
    print(SEP)
    print(f"{name} K-NN Metrics:")
    print(SEP)
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(SEP)


def run_small():
    X_train, X_test, y_train, y_test = train_test_split(
        x_small_dataset_scaled, y_small_dataset,
        test_size=0.3, random_state=RANDOM_STATE, stratify=y_small_dataset,
    )

    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    _print_metrics("Small Dataset", accuracy, precision, recall, f1, roc_auc, elapsed)
    log_knn_result({
        'Dataset': 'Small',
        'Method': 'Baseline',
        'Dimensions': X_train.shape[1],
        'K-Nearest Neighbors Neighbors': N_NEIGHBORS,
        'random_state': RANDOM_STATE,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
        'Time Taken (seconds)': elapsed,
    })


def run_multiclass(name, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)
    elapsed = time.perf_counter() - t0

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    _print_metrics(f"{name} Dataset", accuracy, precision, recall, f1, roc_auc, elapsed)
    log_knn_result({
        'Dataset': name,
        'Method': 'Baseline',
        'Dimensions': X_train.shape[1],
        'K-Nearest Neighbors Neighbors': N_NEIGHBORS,
        'random_state': RANDOM_STATE,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
        'Time Taken (seconds)': elapsed,
    })


if __name__ == "__main__":
    run_small()
    run_multiclass(
        'Medium',
        X_train_medium_dataset, y_train_medium_dataset,
        X_test_medium_dataset, y_test_medium_dataset,
    )
    run_multiclass(
        'Large',
        x_large_dataset_train_scaled, y_large_dataset_train,
        x_large_dataset_test_scaled, y_large_dataset_test,
    )
