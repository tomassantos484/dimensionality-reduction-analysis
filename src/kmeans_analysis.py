# K-Means Analysis (baseline / pre-DR). Single Seed Scenario for testing
#
# Runs k-Means on each dataset at full feature dimensionality and writes one
# row per dataset to results/metrics/practice-runs/kmeans_metrics.csv. The
# multi-seed sweep in src/experiment_runner.py is the canonical entry point;
# this file is kept as a focused, single-seed reference.

import time

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
)

from src.preprocessing import (
    x_small_dataset_scaled, y_small_dataset,
    X_train_medium_dataset, y_train_medium_dataset,
    x_large_dataset_train_scaled, y_large_dataset_train,
)
from src.metric_tracking import log_kmeans_result

N_INIT = 10
RANDOM_STATE = 42
SEP = "=" * 100


def _print_metrics(name, ari, nmi, silhouette, elapsed):
    print(f"{name} K-Means Total Time: {elapsed:.4f}s")
    print(SEP)
    print(f"{name} K-Means Metrics:")
    print(SEP)
    print(f"Adjusted Rand Index (ARI):           {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Silhouette Score (SS):               {silhouette:.4f}")
    print(SEP)


def run_kmeans_baseline(name, X, y, k, ss_sample_size=None):
    t0 = time.perf_counter()
    kmeans = KMeans(
        n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE,
    ).fit(X)
    elapsed = time.perf_counter() - t0
    labels = kmeans.labels_

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    silhouette = silhouette_score(
        X, labels,
        sample_size=ss_sample_size,
        random_state=RANDOM_STATE if ss_sample_size is not None else None,
    )

    _print_metrics(f"{name} Dataset", ari, nmi, silhouette, elapsed)
log_kmeans_result({
        'Dataset': name,
    'Method': 'Baseline',
        'Dimensions': X.shape[1],
        'K-Means Clusters': k,
        'n_init': N_INIT,
        'random_state': RANDOM_STATE,
    'Adjusted Rand Index (ARI)': ari,
    'Normalized Mutual Information (NMI)': nmi,
    'Silhouette Score (SS)': silhouette,
        'SS Sample Size': ss_sample_size,
        'SS Random State': RANDOM_STATE if ss_sample_size is not None else None,
        'Time Taken (seconds)': elapsed,
    })


if __name__ == "__main__":
    run_kmeans_baseline('Small',  x_small_dataset_scaled,       y_small_dataset,       k=2)
    run_kmeans_baseline('Medium', X_train_medium_dataset,       y_train_medium_dataset, k=6)
    run_kmeans_baseline('Large',  x_large_dataset_train_scaled, y_large_dataset_train,
                        k=10, ss_sample_size=10000) # 10K samples is larger than the medium dataset, 60k too slow
