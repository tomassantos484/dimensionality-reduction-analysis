# Principal Component (PCA) Analysis

import time

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Project imports
from preprocessing import (
    x_small_dataset_scaled, y_small_dataset,
    X_train_medium_dataset, y_train_medium_dataset,
    X_test_medium_dataset, y_test_medium_dataset,
    x_large_dataset_train_scaled, y_large_dataset_train,
    x_large_dataset_test_scaled, y_large_dataset_test,
)
from metric_tracking import log_kmeans_result, log_knn_result


# Dimensions to test, 8 comparisons per dataset and algorithm
PCA_DIMENSIONS = [2, 5, 10, 20, 50, 100, 200, 500]


def apply_pca(X_train, d, X_test=None, random_state=42):
    """Fit PCA(d) on X_train. Return (Z_train, Z_test, pca, fit_time).

    Z_test is None when X_test is not provided.
    """
    t0 = time.perf_counter()
    pca = PCA(n_components=d, random_state=random_state)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test) if X_test is not None else None
    fit_time = time.perf_counter() - t0
    return Z_train, Z_test, pca, fit_time


def run_kmeans(Z, y, k, n_init=10, random_state=42, ss_sample_size=None):
    """Fit k-Means on Z and return a metrics dict.

    ss_sample_size=None  -> full silhouette (use for small/medium)
    ss_sample_size=1000  -> sampled silhouette (use for large)
    """
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
    """Fit kNN on Z_train, evaluate on Z_test, return a metrics dict."""
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


def _max_meaningful_d(X):
    """PCA can produce at most min(n_samples, n_features) - 1 components."""
    return min(X.shape[0] - 1, X.shape[1])


def pca_sweep_kmeans(dataset_name, X, y, k,
                     dimensions=PCA_DIMENSIONS,
                     n_init=10, ss_sample_size=None, random_state=42):
    """For each d in dimensions: PCA(d) -> k-Means(k); log each result."""
    max_d = _max_meaningful_d(X)
    print(f"\n[{dataset_name}] PCA + k-Means sweep")
    print("=" * 100)

    for d in dimensions:
        if d > max_d:
            print(f"  Skipping d={d} (max meaningful d here is {max_d})")
            continue

        Z_train, _, pca, pca_time = apply_pca(X, d, random_state=random_state)
        metrics = run_kmeans(
            Z_train, y, k,
            n_init=n_init,
            random_state=random_state,
            ss_sample_size=ss_sample_size,
        )
        # Total time = PCA fit + transform + k-Means fit
        metrics['Time Taken (seconds)'] += pca_time

        log_kmeans_result({
            'Dataset': dataset_name,
            'Method': 'PCA',
            'Dimensions': d,
            **metrics,
        })

        print(
            f"  d={d:>3}  "
            f"ARI={metrics['Adjusted Rand Index (ARI)']:.3f}  "
            f"NMI={metrics['Normalized Mutual Information (NMI)']:.3f}  "
            f"SS={metrics['Silhouette Score (SS)']:.3f}  "
            f"time={metrics['Time Taken (seconds)']:.2f}s  "
            f"(var explained: {pca.explained_variance_ratio_.sum():.3f})"
        )


def pca_sweep_knn(dataset_name, X_train, y_train, X_test, y_test,
                  n_neighbors=5, multiclass=False,
                  dimensions=PCA_DIMENSIONS, random_state=42):
                  
    max_d = _max_meaningful_d(X_train)
    print(f"\n[{dataset_name}] PCA + kNN sweep")
    print("=" * 100)

    for d in dimensions:
        if d > max_d:
            print(f"  Skipping d={d} (max meaningful d here is {max_d})")
            continue

        Z_train, Z_test, pca, pca_time = apply_pca(
            X_train, d, X_test=X_test, random_state=random_state,
        )
        metrics = run_knn(
            Z_train, y_train, Z_test, y_test,
            n_neighbors=n_neighbors, multiclass=multiclass,
        )
        # Total time = PCA fit + transform + kNN fit + predict
        metrics['Time Taken (seconds)'] += pca_time

        log_knn_result({
            'Dataset': dataset_name,
            'Method': 'PCA',
            'Dimensions': d,
            **metrics,
        })

        print(
            f"  d={d:>3}  "
            f"Acc={metrics['Accuracy']:.3f}  "
            f"F1={metrics['F1 Score']:.3f}  "
            f"AUC={metrics['ROC AUC Score']:.3f}  "
            f"time={metrics['Time Taken (seconds)']:.2f}s  "
            f"(var explained: {pca.explained_variance_ratio_.sum():.3f})"
        )


# ----- Small dataset (binary, no pre-defined train/test split) --------
pca_sweep_kmeans(
    'Small',
    x_small_dataset_scaled,
    y_small_dataset,
    k=2,
    ss_sample_size=None,    # tiny dataset, use full silhouette
)

X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(
    x_small_dataset_scaled, y_small_dataset,
    test_size=0.3, random_state=42, stratify=y_small_dataset,
)
pca_sweep_knn(
    'Small',
    X_small_train, y_small_train, X_small_test, y_small_test,
    n_neighbors=5, multiclass=False,
)

# ----- Medium dataset (6 classes, pre-split) --------------------------
pca_sweep_kmeans(
    'Medium',
    X_train_medium_dataset, y_train_medium_dataset,
    k=6,
    ss_sample_size=None,
)
pca_sweep_knn(
    'Medium',
    X_train_medium_dataset, y_train_medium_dataset,
    X_test_medium_dataset,  y_test_medium_dataset,
    n_neighbors=5, multiclass=True,
)

# ----- Large dataset (10 classes, pre-split) --------------------------
pca_sweep_kmeans(
    'Large',
    x_large_dataset_train_scaled, y_large_dataset_train,
    k=10,
    ss_sample_size=1000,    # full silhouette on 60k rows is too slow!
)
pca_sweep_knn(
    'Large',
    x_large_dataset_train_scaled, y_large_dataset_train,
    x_large_dataset_test_scaled,  y_large_dataset_test,
    n_neighbors=5, multiclass=True,
)



print("\nAll PCA sweeps complete! Check results in both results/metrics/kmeans_metrics.csv and results/metrics/knn_metrics.csv")