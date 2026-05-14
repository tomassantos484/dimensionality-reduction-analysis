# Experiment Runner to run hundreds of experiments and save the results to a csv file

# Import libraries
import time
import numpy as np
from sklearn.model_selection import train_test_split

# Preprocessing imports
from src.preprocessing import (
    x_small_dataset_scaled, y_small_dataset,
    X_train_medium_dataset, y_train_medium_dataset,
    X_test_medium_dataset, y_test_medium_dataset,
    x_large_dataset_train_scaled, y_large_dataset_train,
    x_large_dataset_test_scaled, y_large_dataset_test,
)

# PCA Analysis imports
from src.pca_analysis import PCA_DIMENSIONS, apply_pca
from src.random_projection_analysis import apply_random_projection
from src.evaluation import run_kmeans, run_knn

# Metric Tracking imports
from src.metric_tracking import (
    log_kmeans_result, log_knn_result,
    reset_kmeans_metrics, reset_knn_metrics,
)

# Random Seeds for numerous experiments, to ensure reproducibility, and variance
SEEDS = [42, 43, 44, 45, 46]

DATASETS = [
    ('Small',  x_small_dataset_scaled, y_small_dataset, None, None, 2, False, None),
    ('Medium', X_train_medium_dataset, y_train_medium_dataset,
               X_test_medium_dataset,  y_test_medium_dataset, 6, True, None),
    ('Large',  x_large_dataset_train_scaled, y_large_dataset_train,
               x_large_dataset_test_scaled,  y_large_dataset_test, 10, True, None), # 10K samples is larger than the medium dataset, 60k too slow/big
]


# Helper Function to bootstrap the training set for deterministic sampling kNN
def _bootstrap_train(X_tr, y_tr, rng):
    n = len(X_tr)
    idx = rng.integers(0, n, size=n)
    X_bs = X_tr.iloc[idx] if hasattr(X_tr, 'iloc') else X_tr[idx]
    y_bs = y_tr.iloc[idx] if hasattr(y_tr, 'iloc') else y_tr[idx]
    return X_bs, y_bs

# Helper Function to run the experiment trials
def trial(ds, seed):
    name, X, y, X_test, y_test, k, multi, ss = ds
    if X_test is None:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y,
        )
    else:
        X_tr, X_te, y_tr, y_te = X, X_test, y, y_test

    # PCA requires d <= min(n_samples, n_features) - 1, so each branch is
    # bounded by the matrix it actually fits on (X vs. the post-split X_tr).
    max_d_kmeans = min(X.shape) - 1
    max_d_knn = min(X_tr.shape) - 1

    # Loop d outermost so each iteration writes 1 baseline + 1 PCA + 1 RP row
    # per algorithm. This pairs every baseline replica with a DR config and
    # equalizes the per-(Dataset, Method) row counts (~40 rows each at 5 seeds).
    for d in PCA_DIMENSIONS:
        # Per-(seed, d) random state. For baselines this drives independent
        # k-Means inits and kNN train-set bootstraps so each replica is a real
        # new draw, not a duplicate of the seed=outer baseline.
        slot_rs = seed * 1000 + d

        # For baseline kNN and k-Meansreplicas, paired with this d slot
        if d <= max_d_kmeans:
            log_kmeans_result({
                'Dataset': name, 'Method': 'Baseline', 'Dimensions': X.shape[1],
                **run_kmeans(X, y, k, random_state=slot_rs, ss_sample_size=ss),
            })

        if d <= max_d_knn:
            rng = np.random.default_rng(slot_rs)
            X_tr_bs, y_tr_bs = _bootstrap_train(X_tr, y_tr, rng)
            log_knn_result({
                'Dataset': name, 'Method': 'Baseline', 'Dimensions': X.shape[1],
                'random_state': slot_rs,
                **run_knn(X_tr_bs, y_tr_bs, X_te, y_te, multiclass=multi),
            })

        # DR runs at this d
        for method, reduce in [('PCA', apply_pca),
                               ('Random Projection', apply_random_projection)]:
            if d <= max_d_kmeans:
                z, _, reducer, dt = reduce(X, d, random_state=seed)
                m = run_kmeans(z, y, k, random_state=seed, ss_sample_size=ss)
                m['Time Taken (seconds)'] += dt
                # Only PCA exposes explained_variance_ratio_; RP rows leave it blank.
                if hasattr(reducer, 'explained_variance_ratio_'):
                    m['Explained Variance'] = float(reducer.explained_variance_ratio_.sum())
                log_kmeans_result({'Dataset': name, 'Method': method, 'Dimensions': d, **m})

            if d <= max_d_knn:
                z_tr, z_te, _, dt = reduce(X_tr, d, X_test=X_te, random_state=seed)
                m = run_knn(z_tr, y_tr, z_te, y_te, multiclass=multi)
                m['Time Taken (seconds)'] += dt
                log_knn_result({'Dataset': name, 'Method': method, 'Dimensions': d,
                                'random_state': seed, **m})


if __name__ == "__main__":
    reset_kmeans_metrics()
    reset_knn_metrics()
    t0 = time.perf_counter()
    for seed in SEEDS:
        print(f"=== seed {seed} ===")
        for ds in DATASETS:
            ds_t0 = time.perf_counter()
            trial(ds, seed)
            print(f"  {ds[0]}: {time.perf_counter() - ds_t0:.1f}s")
    print(f"Total: {time.perf_counter() - t0:.1f}s ({(time.perf_counter() - t0)/60:.1f} min)")