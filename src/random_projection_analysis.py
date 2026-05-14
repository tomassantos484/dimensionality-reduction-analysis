# Random Projection Analysis — Single Seed Scenario for testing

# Import libraries
import time

from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection

# Preprocessing imports
from src.preprocessing import (
    x_small_dataset_scaled, y_small_dataset,
    X_train_medium_dataset, y_train_medium_dataset,
    X_test_medium_dataset, y_test_medium_dataset,
    x_large_dataset_train_scaled, y_large_dataset_train,
    x_large_dataset_test_scaled, y_large_dataset_test,
)

# Metric Tracking imports
from src.metric_tracking import log_kmeans_result, log_knn_result
from src.evaluation import run_kmeans, run_knn


# Same dimensions as PCA for direct comparison
RP_DIMENSIONS = [2, 5, 10, 20, 50, 100, 200, 500]


def apply_random_projection(X_train, d, X_test=None, random_state=42):
    t0 = time.perf_counter()

    rp = GaussianRandomProjection(
        n_components=d,
        random_state=random_state,
    )

    Z_train = rp.fit_transform(X_train)
    Z_test = rp.transform(X_test) if X_test is not None else None

    fit_time = time.perf_counter() - t0
    return Z_train, Z_test, rp, fit_time

# Helper Function to get the maximum reduction dimension
def _max_reduction_d(X):
    return X.shape[1]

# Random Projection + k-Means sweep
def rp_sweep_kmeans(dataset_name, X, y, k,
                    dimensions=RP_DIMENSIONS,
                    n_init=10,
                    ss_sample_size=None,
                    random_state=42):
    max_d = _max_reduction_d(X)

    print(f"\n[{dataset_name}] Random Projection + k-Means sweep")
    print("=" * 100)

    for d in dimensions:
        if d > max_d:
            print(f"  Skipping d={d} (dataset only has {max_d} features)")
            continue

        Z_train, _, rp, rp_time = apply_random_projection(
            X,
            d,
            random_state=random_state,
        )

        metrics = run_kmeans(
            Z_train,
            y,
            k,
            n_init=n_init,
            random_state=random_state,
            ss_sample_size=ss_sample_size,
        )

        # Total time = RP transform + k-Means fit
        metrics['Time Taken (seconds)'] += rp_time

        log_kmeans_result({
            'Dataset': dataset_name,
            'Method': 'Random Projection',
            'Dimensions': d,
            **metrics,
        })

        print(
            f"  d={d:>3}  "
            f"ARI={metrics['Adjusted Rand Index (ARI)']:.3f}  "
            f"NMI={metrics['Normalized Mutual Information (NMI)']:.3f}  "
            f"SS={metrics['Silhouette Score (SS)']:.3f}  "
            f"time={metrics['Time Taken (seconds)']:.2f}s"
        )

# Random Projection + kNN sweep
def rp_sweep_knn(dataset_name, X_train, y_train, X_test, y_test,
                 n_neighbors=5,
                 multiclass=False,
                 dimensions=RP_DIMENSIONS,
                 random_state=42):
    """For each d: Random Projection(d) -> kNN; log each result."""
    max_d = _max_reduction_d(X_train)

    print(f"\n[{dataset_name}] Random Projection + kNN sweep")
    print("=" * 100)

    for d in dimensions:
        if d > max_d:
            print(f"  Skipping d={d} (dataset only has {max_d} features)")
            continue

        Z_train, Z_test, _, rp_time = apply_random_projection(
            X_train,
            d,
            X_test=X_test,
            random_state=random_state,
        )

        metrics = run_knn(
            Z_train,
            y_train,
            Z_test,
            y_test,
            n_neighbors=n_neighbors,
            multiclass=multiclass,
        )

        # Total time = RP transform + kNN fit + predict
        metrics['Time Taken (seconds)'] += rp_time

        log_knn_result({
            'Dataset': dataset_name,
            'Method': 'Random Projection',
            'Dimensions': d,
            **metrics,
        })

        print(
            f"  d={d:>3}  "
            f"Acc={metrics['Accuracy']:.3f}  "
            f"F1={metrics['F1 Score']:.3f}  "
            f"AUC={metrics['ROC AUC Score']:.3f}  "
            f"time={metrics['Time Taken (seconds)']:.2f}s"
        )


if __name__ == "__main__":

# Small dataset
    rp_sweep_kmeans(
        'Small',
        x_small_dataset_scaled,
        y_small_dataset,
        k=2,
        ss_sample_size=None,
    )

    X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(
        x_small_dataset_scaled,
        y_small_dataset,
        test_size=0.3,
        random_state=42,
        stratify=y_small_dataset,
    )

    rp_sweep_knn(
        'Small',
        X_small_train, y_small_train, X_small_test, y_small_test,
        n_neighbors=5,
        multiclass=False,
    )

    # ----- Medium dataset --------------------------------------------------
    rp_sweep_kmeans(
        'Medium',
        X_train_medium_dataset,
        y_train_medium_dataset,
        k=6,
        ss_sample_size=None,
    )

    rp_sweep_knn(
    'Medium',
    X_train_medium_dataset, y_train_medium_dataset,
    X_test_medium_dataset, y_test_medium_dataset,
    n_neighbors=5,
    multiclass=True,
    )

    # Large dataset
    rp_sweep_kmeans(
        'Large',
        x_large_dataset_train_scaled,
        y_large_dataset_train,
        k=10,
        ss_sample_size=10000, # 10K samples is larger than the medium dataset, 60k too slow/big
    )

    rp_sweep_knn(
        'Large',
        x_large_dataset_train_scaled, y_large_dataset_train,
        x_large_dataset_test_scaled, y_large_dataset_test,
        n_neighbors=5,
        multiclass=True,
    )

    print("\nAll Random Projection sweeps complete! Check results in both results/metrics/kmeans_metrics.csv and results/metrics/knn_metrics.csv")
    
