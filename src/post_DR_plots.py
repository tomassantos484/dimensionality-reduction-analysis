# Plots for post-dimensionality reduction analysis.
#
# Reads the multi-seed sweep CSVs produced by src/experiment_runner.py and
# renders four figures into results/graphs/post-DR:
#   - quality_vs_dim_kmeans.png       (ARI, NMI, Silhouette vs. d)
#   - quality_vs_dim_knn.png          (Accuracy, F1, ROC AUC vs. d)
#   - runtime_vs_dim.png              (log-log runtime curves)
#   - explained_variance_vs_dim.png   (PCA cumulative variance vs. d)

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

KMEANS_CSV = pathlib.Path('results/metrics/practice-runs/kmeans_metrics.csv')
KNN_CSV = pathlib.Path('results/metrics/practice-runs/knn_metrics.csv')
GRAPHS_DIR = pathlib.Path('results/graphs/post-DR')

DATASETS = ['Small', 'Medium', 'Large']
DR_METHODS = ['PCA', 'Random Projection']

# Consistent colors + line styles so the same method always looks the same.
METHOD_STYLE = {
    'PCA':               {'color': '#1f77b4', 'marker': 'o'},
    'Random Projection': {'color': '#d62728', 'marker': 's'},
}
BASELINE_STYLE = {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 1.5}

KMEANS_METRICS = [
    ('Adjusted Rand Index (ARI)',         'ARI'),
    ('Normalized Mutual Information (NMI)', 'NMI'),
    ('Silhouette Score (SS)',             'Silhouette'),
]
KNN_METRICS = [
    ('Accuracy',      'Accuracy'),
    ('F1 Score',      'F1'),
    ('ROC AUC Score', 'ROC AUC'),
]


def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return mean+std of `metric` grouped by (Dataset, Method, Dimensions)."""
    return (
        df.groupby(['Dataset', 'Method', 'Dimensions'], as_index=False)[metric]
          .agg(mean='mean', std='std')
          .fillna({'std': 0.0})
    )


def _plot_quality_panel(ax, agg: pd.DataFrame, dataset: str, metric: str,
                        baseline_value: float | None, title: str) -> None:
    for method in DR_METHODS:
        sub = agg.query("Dataset == @dataset and Method == @method").sort_values('Dimensions')
        if sub.empty:
            continue
        style = METHOD_STYLE[method]
        ax.plot(sub['Dimensions'], sub['mean'],
                label=method, marker=style['marker'], color=style['color'])
        ax.fill_between(sub['Dimensions'],
                        sub['mean'] - sub['std'],
                        sub['mean'] + sub['std'],
                        color=style['color'], alpha=0.18)

    if baseline_value is not None:
        ax.axhline(baseline_value, label=f'Baseline ({baseline_value:.3f})',
                   **BASELINE_STYLE)

    ax.set_xscale('log')
    ax.set_xlabel('Dimensions (d)')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)


def _plot_quality_grid(df: pd.DataFrame, metrics: list[tuple[str, str]],
                       out_path: pathlib.Path, suptitle: str) -> None:
    n_rows, n_cols = len(metrics), len(DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for row, (metric_col, metric_label) in enumerate(metrics):
        agg = _aggregate(df, metric_col)
        baseline_agg = _aggregate(df.query("Method == 'Baseline'"), metric_col)
        for col, dataset in enumerate(DATASETS):
            ax = axes[row][col]
            b = baseline_agg.query("Dataset == @dataset")
            baseline_value = float(b['mean'].iloc[0]) if not b.empty else None
            _plot_quality_panel(ax, agg, dataset, metric_label,
                                baseline_value, f'{dataset} — {metric_label}')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def _plot_runtime(kmeans_df: pd.DataFrame, knn_df: pd.DataFrame,
                  out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, len(DATASETS),
                             figsize=(5 * len(DATASETS), 7),
                             squeeze=False)

    for row, (df, algo_label) in enumerate([(kmeans_df, 'k-Means'), (knn_df, 'kNN')]):
        agg = _aggregate(df, 'Time Taken (seconds)')
        baseline_agg = _aggregate(df.query("Method == 'Baseline'"),
                                  'Time Taken (seconds)')
        for col, dataset in enumerate(DATASETS):
            ax = axes[row][col]
            for method in DR_METHODS:
                sub = (agg.query("Dataset == @dataset and Method == @method")
                          .sort_values('Dimensions'))
                if sub.empty:
                    continue
                style = METHOD_STYLE[method]
                # Timing variance is dominated by GC/cache noise; just plot the
                # seed-mean to keep the speedup signal readable on log-log.
                ax.plot(sub['Dimensions'], sub['mean'],
                        label=method, marker=style['marker'], color=style['color'])

            b = baseline_agg.query("Dataset == @dataset")
            if not b.empty:
                baseline_value = float(b['mean'].iloc[0])
                ax.axhline(baseline_value,
                           label=f'Baseline ({baseline_value:.2f}s)',
                           **BASELINE_STYLE)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Dimensions (d)')
            ax.set_ylabel('Time (s)')
            ax.set_title(f'{dataset} — {algo_label}')
            ax.grid(True, which='both', alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Runtime vs. Dimensionality (log-log)', fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


DATASET_COLORS = {
    'Small':  '#4c72b0',
    'Medium': '#dd8452',
    'Large':  '#55a467',
}


def _plot_explained_variance(kmeans_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """One line per dataset: cumulative PCA explained variance vs. d.

    The CSV must have been produced by an experiment_runner run that postdates
    the introduction of the 'Explained Variance' column; otherwise we skip
    with a friendly message rather than crashing.
    """
    if 'Explained Variance' not in kmeans_df.columns:
        print(f'  skip {out_path.name}: '
              "kmeans_metrics.csv has no 'Explained Variance' column. "
              'Re-run `python -m src.experiment_runner` to populate it.')
        return

    pca = kmeans_df.query("Method == 'PCA'").dropna(subset=['Explained Variance'])
    if pca.empty:
        print(f'  skip {out_path.name}: no PCA rows with explained variance.')
        return

    agg = (pca.groupby(['Dataset', 'Dimensions'], as_index=False)['Explained Variance']
              .agg(mean='mean', std='std')
              .fillna({'std': 0.0}))

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        sub = agg.query("Dataset == @dataset").sort_values('Dimensions')
        if sub.empty:
            continue
        color = DATASET_COLORS.get(dataset, None)
        ax.plot(sub['Dimensions'], sub['mean'],
                marker='o', label=dataset, color=color)
        ax.fill_between(sub['Dimensions'],
                        sub['mean'] - sub['std'],
                        sub['mean'] + sub['std'],
                        alpha=0.18, color=color)

    ax.axhline(0.9, linestyle=':', color='gray', alpha=0.6, label='90% variance')
    ax.set_xscale('log')
    ax.set_xlabel('Dimensions (d)')
    ax.set_ylabel('Cumulative explained variance ratio')
    ax.set_title('PCA cumulative explained variance vs. d')
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right')

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
                       GRAPHS_DIR / 'quality_vs_dim_kmeans.png',
                       'k-Means Clustering Quality vs. PCA / Random Projection Dimension')
    _plot_quality_grid(knn_df, KNN_METRICS,
                       GRAPHS_DIR / 'quality_vs_dim_knn.png',
                       'kNN Classification Quality vs. PCA / Random Projection Dimension')
    _plot_runtime(kmeans_df, knn_df, GRAPHS_DIR / 'runtime_vs_dim.png')
    _plot_explained_variance(kmeans_df, GRAPHS_DIR / 'explained_variance_vs_dim.png')


if __name__ == '__main__':
    make_all_plots()
