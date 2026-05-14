# Plots for pre-dimensionality reduction (baseline) analysis.

# Reads the same multi-seed sweep CSVs as post_DR_plots.py, but slices to the
# Method == 'Baseline' rows only. Each baseline configuration is a single point
# in dimension space (the full feature count), so we render grouped bar charts
# with ±1σ error bars rather than curves.

# Output figures (results/graphs/):
#   - baseline_quality_kmeans.png  (ARI, NMI, Silhouette per dataset)
#   - baseline_quality_knn.png     (Accuracy, F1, ROC AUC per dataset)
#   - baseline_runtime.png         (log-y bar chart of k-Means / kNN time)

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KMEANS_CSV = pathlib.Path('results/metrics/practice-runs/kmeans_metrics.csv')
KNN_CSV = pathlib.Path('results/metrics/practice-runs/knn_metrics.csv')
GRAPHS_DIR = pathlib.Path('results/graphs/pre-DR')

DATASETS = ['Small', 'Medium', 'Large']

# Utilities for plotting
DATASET_COLORS = {
    'Small':  '#4c72b0',
    'Medium': '#dd8452',
    'Large':  '#55a467',
}

KMEANS_METRICS = [
    ('Adjusted Rand Index (ARI)',           'ARI'),
    ('Normalized Mutual Information (NMI)', 'NMI'),
    ('Silhouette Score (SS)',               'Silhouette'),
]
KNN_METRICS = [
    ('Accuracy',      'Accuracy'),
    ('F1 Score',      'F1'),
    ('ROC AUC Score', 'ROC AUC'),
]


def _baseline_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    baseline = df.query("Method == 'Baseline'")
    return (
        baseline.groupby('Dataset', as_index=False)[metric]
                .agg(mean='mean', std='std')
                .fillna({'std': 0.0})
    )


def _plot_metric_bars(ax, df: pd.DataFrame, metric_col: str, metric_label: str) -> None:
    stats = _baseline_stats(df, metric_col)
    stats = stats.set_index('Dataset').reindex(DATASETS).reset_index()

    x = np.arange(len(DATASETS))
    means = stats['mean'].to_numpy()
    stds = stats['std'].to_numpy()
    colors = [DATASET_COLORS[d] for d in DATASETS]

    bars = ax.bar(x, means, yerr=stds, color=colors,
                  capsize=4, edgecolor='black', linewidth=0.6)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x, DATASETS)
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.grid(True, axis='y', alpha=0.3)
    # Give the value labels headroom.
    top = max(means + stds) if len(means) else 1.0
    ax.set_ylim(0, top * 1.15)


def _plot_quality_grid(df: pd.DataFrame, metrics: list[tuple[str, str]],
                       out_path: pathlib.Path, suptitle: str) -> None:
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(5 * len(metrics), 4),
                             squeeze=False)
    for col, (metric_col, metric_label) in enumerate(metrics):
        _plot_metric_bars(axes[0][col], df, metric_col, metric_label)

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def _plot_baseline_runtime(kmeans_df: pd.DataFrame, knn_df: pd.DataFrame,
                           out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    kmeans_stats = (_baseline_stats(kmeans_df, 'Time Taken (seconds)')
                    .set_index('Dataset').reindex(DATASETS).reset_index())
    knn_stats = (_baseline_stats(knn_df, 'Time Taken (seconds)')
                 .set_index('Dataset').reindex(DATASETS).reset_index())

    x = np.arange(len(DATASETS))
    width = 0.38

    kmeans_bars = ax.bar(x - width / 2, kmeans_stats['mean'], width,
                         yerr=kmeans_stats['std'], capsize=4,
                         label='k-Means', color='#4c72b0', edgecolor='black',
                         linewidth=0.6)
    knn_bars = ax.bar(x + width / 2, knn_stats['mean'], width,
                      yerr=knn_stats['std'], capsize=4,
                      label='kNN', color='#dd8452', edgecolor='black',
                      linewidth=0.6)

    for bars, means in [(kmeans_bars, kmeans_stats['mean']),
                        (knn_bars, knn_stats['mean'])]:
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value:.2f}s', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x, DATASETS)
    ax.set_yscale('log')
    ax.set_ylabel('Time (s, log scale)')
    ax.set_title('Baseline runtime (full feature dimensionality)')
    ax.grid(True, axis='y', which='both', alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def make_all_plots() -> None:
    if not KMEANS_CSV.exists() or not KNN_CSV.exists():
        raise FileNotFoundError(
            f'Missing metric CSVs. Expected:\n  {KMEANS_CSV}\n  {KNN_CSV}\n'
            'Run `python -m src.experiment_runner` first.'
        )

    kmeans_df = pd.read_csv(KMEANS_CSV)
    knn_df = pd.read_csv(KNN_CSV)

    _plot_quality_grid(kmeans_df, KMEANS_METRICS,
                       GRAPHS_DIR / 'baseline_quality_kmeans.png',
                       'k-Means Baseline Quality (pre-DR, full dimensionality)')
    _plot_quality_grid(knn_df, KNN_METRICS,
                       GRAPHS_DIR / 'baseline_quality_knn.png',
                       'kNN Baseline Quality (pre-DR, full dimensionality)')
    _plot_baseline_runtime(kmeans_df, knn_df, GRAPHS_DIR / 'baseline_runtime.png')


if __name__ == '__main__':
    make_all_plots()
