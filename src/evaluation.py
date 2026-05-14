# Shared evaluation helpers.
#
# Single source of truth for the per-(dataset, method, d) fit-and-score logic
# used by every DR sweep (baseline, PCA, Random Projection). Imported by
# src.pca_analysis, src.random_projection_analysis, and src.experiment_runner
# so the same numbers come out regardless of caller.

import time

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier


def run_kmeans(Z, y, k, n_init=10, random_state=42, ss_sample_size=None):
    t0 = time.perf_counter()
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Z)
    fit_time = time.perf_counter() - t0
    labels = kmeans.labels_

    silhouette = silhouette_score(
        Z, labels,
        sample_size=ss_sample_size,
        random_state=random_state if ss_sample_size is not None else None,
    )

    return {
        'K-Means Clusters': k,
        'n_init': n_init,
        'random_state': random_state,
        'Adjusted Rand Index (ARI)': adjusted_rand_score(y, labels),
        'Normalized Mutual Information (NMI)': normalized_mutual_info_score(y, labels),
        'Silhouette Score (SS)': silhouette,
        'SS Sample Size': ss_sample_size,
        'SS Random State': random_state if ss_sample_size is not None else None,
        'Time Taken (seconds)': fit_time,
    }


def run_knn(Z_train, y_train, Z_test, y_test, n_neighbors=5, multiclass=False):
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(Z_train, y_train)
    y_pred = knn.predict(Z_test)
    y_proba = knn.predict_proba(Z_test)
    fit_time = time.perf_counter() - t0

    if multiclass:
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall    = recall_score(y_test,    y_pred, average='macro')
        f1        = f1_score(y_test,        y_pred, average='macro')
        roc_auc   = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    else:
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test,    y_pred)
        f1        = f1_score(y_test,        y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba[:, 1])

    return {
        'K-Nearest Neighbors Neighbors': n_neighbors,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
        'Time Taken (seconds)': fit_time,
    }
