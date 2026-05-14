# Silhouette Test to test the silhouette score at varying sample sizes

import time

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing import x_large_dataset_train_scaled

RANDOM_STATE = 42
K = 10
SAMPLE_SIZES = [1000, 5000, 10000, 30000, None]  # None = full 60,000 rows


def main() -> None:
    X = x_large_dataset_train_scaled.to_numpy()
    print(f"Large dataset shape: {X.shape}")

    # Fit k-Means once so every silhouette call scores the same labels — the
    # only variable in this experiment is the sample_size argument.
    print("\nFitting k-Means(k=10) on full Large dataset...")
    t0 = time.perf_counter()
    labels = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(X).labels_
    print(f"  k-Means fit time: {time.perf_counter() - t0:.2f}s")

    rows = []
    print("\nTiming silhouette_score at varying sample_size:")
    for n in SAMPLE_SIZES:
        t0 = time.perf_counter()
        score = silhouette_score(X, labels, sample_size=n, random_state=RANDOM_STATE)
        elapsed = time.perf_counter() - t0
        label = 'full (60000)' if n is None else f'{n}'
        print(f"  sample_size={label:>14}  silhouette={score:.4f}  time={elapsed:7.2f}s")
        rows.append({'sample_size': label, 'silhouette': round(score, 4),
                     'time_s': round(elapsed, 3)})


if __name__ == "__main__":
    main()