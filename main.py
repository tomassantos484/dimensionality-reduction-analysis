# Main Program

# Import libraries
import os

import pandas as pd

from sklearn.model_selection import train_test_split


# Preprocessing imports
from src.preprocessing import (
    x_small_dataset_scaled, y_small_dataset,
    X_train_medium_dataset, y_train_medium_dataset,
    X_test_medium_dataset, y_test_medium_dataset,
    x_large_dataset_train_scaled, y_large_dataset_train,
    x_large_dataset_test_scaled, y_large_dataset_test,
)

# Shared evaluation helpers (used by the baseline live demo so it produces
# numbers comparable to the PCA / RP live demos).
from src.evaluation import run_kmeans, run_knn

# PCA Analysis imports
from src.pca_analysis import (
    PCA_DIMENSIONS,
    apply_pca,
    run_kmeans as run_pca_kmeans,
    run_knn as run_pca_knn
)

# Random Projection Analysis imports
from src.random_projection_analysis import (
    RP_DIMENSIONS,
    apply_random_projection,
    run_kmeans as run_rp_kmeans,
    run_knn as run_rp_knn
)

# Metric Tracking imports (practice runs vs. live demo runs)
from src.metric_tracking import (
    KNN_METRICS_PATH_PRACTICE,
    KMEANS_METRICS_PATH_PRACTICE,
    KNN_METRICS_PATH_LIVE_DEMO,
    KMEANS_METRICS_PATH_LIVE_DEMO,
    log_kmeans_demo_result,
    log_knn_demo_result,
)

# Utility Functions

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause_for_input():
    input("\nPress Enter to continue...")

def print_title():
    print("="*100)
    print("Welcome to the Dimensionality Reduction Program!")
    print("="*100)

def print_100_equals():
    print("="*100)

def max_pca_dimension(X):
    return min(X.shape[0] - 1, X.shape[1])

def max_rp_dimension(X):
    # Random Projection has no min(n-1, p) cap like PCA does, but projecting
    # to more than the original feature count defeats the point of reducing.
    return X.shape[1]

def user_menu():
    clear_screen()
    print_title()
    print("Please select an option:")
    print("1. Dataset Overview")
    print("2. Baseline Analysis (Pre-Dimensionality Reduction)")
    print("3. Principal Component Analysis (PCA) Results")
    print("4. Random Projection Results")
    print("5. Comparison of Dimensionality Reduction Results")
    print("0. Exit")
    print_100_equals()

    return input("\nEnter your choice: ")


def dispatch_choice(choice):
    user_actions = {
        '1': dataset_overview,
        '2': baseline_analysis,
        '3': pca_results,
        '4': random_projection_results,
        '5': comparison_of_dimensionality_reduction_results,
    }

    if choice in user_actions:
        user_actions[choice]()
    elif choice == '0':
        return
    else:
        print("Invalid choice, please try again.")
        pause_for_input()


# Actual Demo Content

def dataset_overview():
    clear_screen()
    print_title()
    print("Dataset Overview: Meet your datasets!")
    print_100_equals()

    rows = [
        ('Small Dataset (High-Dimensionality Dataset — Kaggle)', x_small_dataset_scaled.shape, 2),
        ('Medium Dataset (UCI HAR Dataset)', X_train_medium_dataset.shape, 6),
        ('Large Dataset (Fashion-MNIST)', x_large_dataset_train_scaled.shape, 10),
    ]

    # Turn into pd df
    df = pd.DataFrame(rows, columns=['Dataset', 'Shape', '# of Classes'])
    print(df)


def baseline_analysis():
    while True:
        clear_screen()
        print_title()
        print("Baseline Analysis: pre-dimensionality-reduction reference points")

        print_100_equals()
        print("Please select option:")
        print("1. Baseline Summary Table (Practice Runs)")
        print("2. Baseline Live Demo (pick dataset, algorithm)")
        print("0. Return to Main Menu")
        print_100_equals()

        choice = input("\nEnter your choice: ")
        if choice == '1':
            baseline_summary_table()
            pause_for_input()
        elif choice == '2':
            baseline_live_demo()
            pause_for_input()
        elif choice == '0':
            return
        else:
            print("Invalid choice, please try again.")
            pause_for_input()


def baseline_summary_table():
    clear_screen()
    print_title()
    print("Baseline Summary Table: Practice Runs (full-dimensionality reference)")
    print_100_equals()

    kmeans_summary = pd.read_csv(KMEANS_METRICS_PATH_PRACTICE).query("Method == 'Baseline'")
    knn_summary    = pd.read_csv(KNN_METRICS_PATH_PRACTICE).query("Method == 'Baseline'")

    # Average across the (seed, dim-slot) replicas emitted by experiment_runner.
    kmeans_summary_table = kmeans_summary.groupby('Dataset').agg(
        d=('Dimensions', 'first'),
        n_runs=('Dimensions', 'count'),
        ARI=('Adjusted Rand Index (ARI)', 'mean'),
        NMI=('Normalized Mutual Information (NMI)', 'mean'),
        Silhouette=('Silhouette Score (SS)', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
    ).round(5).reset_index()

    table_order = pd.Categorical(kmeans_summary_table['Dataset'], categories=['Small', 'Medium', 'Large'])
    kmeans_summary_table = kmeans_summary_table.assign(_order=table_order).sort_values('_order').drop(columns='_order')

    print("\nk-Means Summary Table (averaged across seeds):")
    print(kmeans_summary_table.to_string(index=False))

    print_100_equals()

    knn_summary_table = knn_summary.groupby('Dataset').agg(
        d=('Dimensions', 'first'),
        n_runs=('Dimensions', 'count'),
        Accuracy=('Accuracy', 'mean'),
        Precision=('Precision', 'mean'),
        Recall=('Recall', 'mean'),
        F1=('F1 Score', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
    ).round(5).reset_index()

    table_order = pd.Categorical(knn_summary_table['Dataset'], categories=['Small', 'Medium', 'Large'])
    knn_summary_table = knn_summary_table.assign(_order=table_order).sort_values('_order').drop(columns='_order')

    print("\nk-NN Summary Table (averaged across seeds):")
    print(knn_summary_table.to_string(index=False))
    print_100_equals()


def baseline_live_demo():
    clear_screen()
    print_title()
    print("Baseline Live Demo: run k-Means / kNN at full dimensionality")
    print_100_equals()

    # Same dataset_options shape as the PCA / RP live demos so the three live
    # menus look and feel identical to the audience.
    dataset_options = {
        '1': {
            'name': 'Small',
            'description': 'High-Dimensionality Dataset — Kaggle',
            'dimensions': x_small_dataset_scaled.shape[1],
            'classes': 2,
            'multiclass': False,
            'kmeans_X': x_small_dataset_scaled,
            'kmeans_y': y_small_dataset,
            'ss_sample_size': None,
        },
        '2': {
            'name': 'Medium',
            'description': 'UCI HAR Dataset',
            'dimensions': X_train_medium_dataset.shape[1],
            'classes': 6,
            'multiclass': True,
            'kmeans_X': X_train_medium_dataset,
            'kmeans_y': y_train_medium_dataset,
            'knn_train_X': X_train_medium_dataset,
            'knn_train_y': y_train_medium_dataset,
            'knn_test_X': X_test_medium_dataset,
            'knn_test_y': y_test_medium_dataset,
            'ss_sample_size': None,
        },
        '3': {
            'name': 'Large',
            'description': 'Fashion-MNIST',
            'dimensions': x_large_dataset_train_scaled.shape[1],
            'classes': 10,
            'multiclass': True,
            'kmeans_X': x_large_dataset_train_scaled,
            'kmeans_y': y_large_dataset_train,
            'knn_train_X': x_large_dataset_train_scaled,
            'knn_train_y': y_large_dataset_train,
            'knn_test_X': x_large_dataset_test_scaled,
            'knn_test_y': y_large_dataset_test,
            'ss_sample_size': 1000,
        },
    }

    print("Choose a dataset:")
    print("1. Small Dataset")
    print("2. Medium Dataset")
    print("3. Large Dataset")
    dataset_choice = input("\nEnter your choice: ")

    if dataset_choice not in dataset_options:
        print("Invalid dataset choice.")
        return

    config = dataset_options[dataset_choice]
    dataset_name = config['name']

    print_100_equals()
    print("Choose an algorithm:")
    print("1. Baseline k-Means")
    print("2. Baseline kNN")
    print("3. Run both")
    algorithm_choice = input("\nEnter your choice: ")

    if algorithm_choice not in {'1', '2', '3'}:
        print("Invalid algorithm choice.")
        return

    print_100_equals()
    print(f"Running baseline on {dataset_name} at full dimensionality "
          f"(d={config['dimensions']})...")

    if algorithm_choice in {'1', '3'}:
        metrics = run_kmeans(
            config['kmeans_X'],
            config['kmeans_y'],
            k=config['classes'],
            ss_sample_size=config['ss_sample_size'],
            random_state=42,
        )

        log_kmeans_demo_result({
            'Dataset': dataset_name,
            'Method': 'Baseline',
            'Dimensions': config['dimensions'],
            **metrics,
        })

        kmeans_table = pd.DataFrame([{
            'Dataset': dataset_name,
            'Dimensions': config['dimensions'],
            'ARI': metrics['Adjusted Rand Index (ARI)'],
            'NMI': metrics['Normalized Mutual Information (NMI)'],
            'Silhouette': metrics['Silhouette Score (SS)'],
            'Time_s': metrics['Time Taken (seconds)'],
        }]).round(4)

        print("\nBaseline k-Means Live Results:")
        print(kmeans_table.to_string(index=False))

    if algorithm_choice in {'2', '3'}:
        # Same Small-dataset split policy as the PCA / RP live demos: held-out
        # 30% with the same seed and stratification, so all three live demos
        # report numbers on the same test split.
        if dataset_name == 'Small':
            X_train, X_test, y_train, y_test = train_test_split(
                x_small_dataset_scaled,
                y_small_dataset,
                test_size=0.3,
                random_state=42,
                stratify=y_small_dataset,
            )
        else:
            X_train = config['knn_train_X']
            y_train = config['knn_train_y']
            X_test  = config['knn_test_X']
            y_test  = config['knn_test_y']

        metrics = run_knn(
            X_train,
            y_train,
            X_test,
            y_test,
            n_neighbors=5,
            multiclass=config['multiclass'],
        )

        log_knn_demo_result({
            'Dataset': dataset_name,
            'Method': 'Baseline',
            'Dimensions': X_train.shape[1],
            **metrics,
        })

        knn_table = pd.DataFrame([{
            'Dataset': dataset_name,
            'Dimensions': X_train.shape[1],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1 Score'],
            'AUC': metrics['ROC AUC Score'],
            'Time_s': metrics['Time Taken (seconds)'],
        }]).round(4)

        print("\nBaseline kNN Live Results:")
        print(knn_table.to_string(index=False))

    print_100_equals()
    print("Live demo complete. Results were logged to the demo-runs CSV files. "
          "Please check the results/metrics/demo-runs directory.")


def pca_results():
    while True:
        clear_screen()
        print_title()
        print("PCA Results: Meet your PCA results!")

        print_100_equals()
        print("Please select option:")
        print("1. PCA Summary Table (Practice Runs)")
        print("2. PCA Live Demo (pick dataset, algorithm, dimensions)")
        print("3. PCA vs. baseline analysis")
        print("0. Return to Main Menu")
        print_100_equals()

        choice = input("\nEnter your choice: ")
        if choice == '1':
            pca_summary_table()
            pause_for_input()
        elif choice == '2':
            pca_live_demo()
            pause_for_input()
        elif choice == '3':
            pca_vs_baseline()
            pause_for_input()
        elif choice == '0':
            return
        else:
            print("Invalid choice, please try again.")
            pause_for_input()

def pca_summary_table():
    clear_screen()
    print_title()
    print("PCA Summary Table: Practice Runs (broken down by dimension)")
    print_100_equals()

    pca_kmeans_summary = pd.read_csv(KMEANS_METRICS_PATH_PRACTICE).query("Method == 'PCA'")
    pca_knn_summary = pd.read_csv(KNN_METRICS_PATH_PRACTICE).query("Method == 'PCA'")

    # Group by (Dataset, Dimensions) so each PCA target d gets its own row,
    # with metrics averaged across however many times that config was run.
    kmeans_by_d = pca_kmeans_summary.groupby(['Dataset', 'Dimensions']).agg(
        ARI=('Adjusted Rand Index (ARI)', 'mean'),
        NMI=('Normalized Mutual Information (NMI)', 'mean'),
        Silhouette=('Silhouette Score (SS)', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
    ).round(4).reset_index()

    knn_by_d = pca_knn_summary.groupby(['Dataset', 'Dimensions']).agg(
        Accuracy=('Accuracy', 'mean'),
        F1=('F1 Score', 'mean'),
        AUC=('ROC AUC Score', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
    ).round(4).reset_index()

    # One pair of tables per dataset so each set of dimensions stays together.
    for dataset in ['Small', 'Medium', 'Large']:
        print(f"\n[{dataset}] PCA + k-Means (one row per dimension)")
        rows = kmeans_by_d.query("Dataset == @dataset").sort_values('Dimensions')
        if rows.empty:
            print("  (no PCA k-Means runs logged for this dataset)")
        else:
            print(rows.drop(columns=['Dataset']).to_string(index=False))

        print(f"\n[{dataset}] PCA + kNN (one row per dimension)")
        rows = knn_by_d.query("Dataset == @dataset").sort_values('Dimensions')
        if rows.empty:
            print("  (no PCA kNN runs logged for this dataset)")
        else:
            print(rows.drop(columns=['Dataset']).to_string(index=False))

        print_100_equals()

def pca_live_demo():
    clear_screen()
    print_title()
    print("PCA Live Demo: run PCA now and evaluate it immediately")
    print_100_equals()

    dataset_options = {
        '1': {
            'name': 'Small',
            'description': 'High-Dimensionality Dataset — Kaggle',
            'dimensions': x_small_dataset_scaled.shape[1],
            'classes': 2,
            'multiclass': False,
            'kmeans_X': x_small_dataset_scaled,
            'kmeans_y': y_small_dataset,
            'ss_sample_size': None,
        },
        '2': {
            'name': 'Medium',
            'description': 'UCI HAR Dataset',
            'dimensions': X_train_medium_dataset.shape[1],
            'classes': 6,
            'multiclass': True,
            'kmeans_X': X_train_medium_dataset,
            'kmeans_y': y_train_medium_dataset,
            'knn_train_X': X_train_medium_dataset,
            'knn_train_y': y_train_medium_dataset,
            'knn_test_X': X_test_medium_dataset,
            'knn_test_y': y_test_medium_dataset,
            'ss_sample_size': None,
        },
        '3': {
            'name': 'Large',
            'description': 'Fashion-MNIST',
            'dimensions': x_large_dataset_train_scaled.shape[1],
            'classes': 10,
            'multiclass': True,
            'kmeans_X': x_large_dataset_train_scaled,
            'kmeans_y': y_large_dataset_train,
            'knn_train_X': x_large_dataset_train_scaled,
            'knn_train_y': y_large_dataset_train,
            'knn_test_X': x_large_dataset_test_scaled,
            'knn_test_y': y_large_dataset_test,
            'ss_sample_size': 1000,
        },
    }

    print("Choose a dataset:")
    print("1. Small Dataset")
    print("2. Medium Dataset")
    print("3. Large Dataset")
    dataset_choice = input("\nEnter your choice: ")

    if dataset_choice not in dataset_options:
        print("Invalid dataset choice.")
        return

    config = dataset_options[dataset_choice]
    dataset_name = config['name']

    print_100_equals()
    print("Choose an algorithm:")
    print("1. PCA + k-Means")
    print("2. PCA + kNN")
    print("3. Run both")
    algorithm_choice = input("\nEnter your choice: ")

    if algorithm_choice not in {'1', '2', '3'}:
        print("Invalid algorithm choice.")
        return

    knn_data = None
    max_dimensions = []

    if algorithm_choice in {'1', '3'}:
        max_dimensions.append(max_pca_dimension(config['kmeans_X']))

    if algorithm_choice in {'2', '3'}:
        if dataset_name == 'Small':
            knn_data = train_test_split(
                x_small_dataset_scaled,
                y_small_dataset,
                test_size=0.3,
                random_state=42,
                stratify=y_small_dataset,
            )
            max_dimensions.append(max_pca_dimension(knn_data[0]))
        else:
            max_dimensions.append(max_pca_dimension(config['knn_train_X']))

    max_d = min(max_dimensions)
    suggested_dimensions = [d for d in PCA_DIMENSIONS if d <= max_d]

    print_100_equals()
    print(f"{dataset_name} has {config['kmeans_X'].shape[1]} original dimensions.")
    print(f"Suggested PCA dimensions: {suggested_dimensions}")
    dimension_choice = input(f"Enter PCA target dimension d (1 to {max_d}): ")

    try:
        d = int(dimension_choice)
    except ValueError:
        print("Dimension must be a whole number.")
        return

    if d < 1 or d > max_d:
        print(f"Invalid dimension. For {dataset_name}, choose a value from 1 to {max_d}.")
        return

    print_100_equals()
    print(f"Running PCA live demo for {dataset_name} with d={d}...")

    if algorithm_choice in {'1', '3'}:
        z_train, _, pca, pca_time = apply_pca(
            config['kmeans_X'],
            d,
            random_state=42,
        )
        metrics = run_pca_kmeans(
            z_train,
            config['kmeans_y'],
            k=config['classes'],
            ss_sample_size=config['ss_sample_size'],
            random_state=42,
        )
        metrics['Time Taken (seconds)'] += pca_time

        log_kmeans_demo_result({
            'Dataset': dataset_name,
            'Method': 'PCA',
            'Dimensions': d,
            **metrics,
        })

        kmeans_table = pd.DataFrame([{
            'Dataset': dataset_name,
            'Dimensions': d,
            'Explained Variance': pca.explained_variance_ratio_.sum(),
            'ARI': metrics['Adjusted Rand Index (ARI)'],
            'NMI': metrics['Normalized Mutual Information (NMI)'],
            'Silhouette': metrics['Silhouette Score (SS)'],
            'Time_s': metrics['Time Taken (seconds)'],
        }]).round(4)

        print("\nPCA + k-Means Live Results:")
        print(kmeans_table.to_string(index=False))

    if algorithm_choice in {'2', '3'}:
        if knn_data is not None:
            X_train, X_test, y_train, y_test = knn_data
        else:
            X_train = config['knn_train_X']
            y_train = config['knn_train_y']
            X_test = config['knn_test_X']
            y_test = config['knn_test_y']

        z_train, z_test, pca, pca_time = apply_pca(
            X_train,
            d,
            X_test=X_test,
            random_state=42,
        )
        metrics = run_pca_knn(
            z_train,
            y_train,
            z_test,
            y_test,
            n_neighbors=5,
            multiclass=config['multiclass'],
        )
        metrics['Time Taken (seconds)'] += pca_time

        log_knn_demo_result({
            'Dataset': dataset_name,
            'Method': 'PCA',
            'Dimensions': d,
            **metrics,
        })

        knn_table = pd.DataFrame([{
            'Dataset': dataset_name,
            'Dimensions': d,
            'Explained Variance': pca.explained_variance_ratio_.sum(),
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1 Score'],
            'AUC': metrics['ROC AUC Score'],
            'Time_s': metrics['Time Taken (seconds)'],
        }]).round(4)

        print("\nPCA + kNN Live Results:")
        print(knn_table.to_string(index=False))

    print_100_equals()
    print("Live demo complete. Results were logged to the demo-runs CSV files. Please check the results/metrics/demo-runs directory.")


def pca_vs_baseline():
    clear_screen()
    print_title()
    print("PCA vs. Baseline: how much do we gain (or lose) by reducing dimensions?")
    print_100_equals()

    kmeans_all = pd.read_csv(KMEANS_METRICS_PATH_PRACTICE).query("Method in ['Baseline', 'PCA']")
    knn_all    = pd.read_csv(KNN_METRICS_PATH_PRACTICE).query("Method in ['Baseline', 'PCA']")

    # Aggregate repeated rows so each (Dataset, Method, Dimensions) is unique.
    # We also keep a count so we can flag baselines that came from multiple
    # recordings (e.g. Large baseline timed under several silhouette sample sizes).
    kmeans_agg = kmeans_all.groupby(['Dataset', 'Method', 'Dimensions']).agg(
        ARI=('Adjusted Rand Index (ARI)', 'mean'),
        NMI=('Normalized Mutual Information (NMI)', 'mean'),
        Silhouette=('Silhouette Score (SS)', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
        n_runs=('Time Taken (seconds)', 'count'),
    ).reset_index()

    knn_agg = knn_all.groupby(['Dataset', 'Method', 'Dimensions']).agg(
        Accuracy=('Accuracy', 'mean'),
        F1=('F1 Score', 'mean'),
        AUC=('ROC AUC Score', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
        n_runs=('Time Taken (seconds)', 'count'),
    ).reset_index()

    for dataset in ['Small', 'Medium', 'Large']:
        _print_method_vs_baseline_for_algo(
            dataset,
            kmeans_agg.query("Dataset == @dataset"),
            algorithm_label='PCA + k-Means',
            quality_col='ARI',
            display_cols=['Method', 'Dimensions', 'ARI', 'NMI', 'Silhouette', 'Time_s'],
        )

        _print_method_vs_baseline_for_algo(
            dataset,
            knn_agg.query("Dataset == @dataset"),
            algorithm_label='PCA + kNN',
            quality_col='Accuracy',
            display_cols=['Method', 'Dimensions', 'Accuracy', 'F1', 'AUC', 'Time_s'],
        )

        print_100_equals()


def _print_method_vs_baseline_for_algo(dataset, rows, algorithm_label, quality_col, display_cols):
    print(f"\n[{dataset}] {algorithm_label} vs. Baseline")

    if rows.empty:
        print("  (no runs logged for this dataset)")
        return

    rows = rows.copy()
    baseline = rows.query("Method == 'Baseline'")

    if baseline.empty:
        print("  (no baseline row found; showing reduction rows only)")
        rows = rows.sort_values('Dimensions')
        print(rows[display_cols].round(4).to_string(index=False))
        return

    # Flag aggregated baselines so the audience knows that the time column
    # for that row is a mean of multiple recordings, not a single timing.
    baseline_run_count = int(baseline['n_runs'].iloc[0])
    if baseline_run_count > 1:
        print(
            f"  (baseline row is averaged across {baseline_run_count} recorded runs; "
            f"timings may not reflect a single configuration)"
        )

    baseline_quality = baseline[quality_col].iloc[0]
    baseline_time = baseline['Time_s'].iloc[0]
    rows[f'Δ{quality_col}'] = rows[quality_col] - baseline_quality
    rows['ΔTime_s'] = rows['Time_s'] - baseline_time

    # Order: Baseline first, then DR rows by dimension ascending.
    rows['_order'] = (rows['Method'] != 'Baseline').astype(int)
    rows = rows.sort_values(['_order', 'Dimensions']).drop(columns=['_order'])

    columns_to_show = display_cols + [f'Δ{quality_col}', 'ΔTime_s']
    print(rows[columns_to_show].round(4).to_string(index=False))


def random_projection_results():
    while True:
        clear_screen()
        print_title()
        print("Random Projection Results: Meet your Random Projection results!")

        print_100_equals()
        print("Please select option:")
        print("1. Random Projection Summary Table (Practice Runs)")
        print("2. Random Projection Live Demo (pick dataset, algorithm, dimensions)")
        print("3. Random Projection vs. baseline analysis")
        print("0. Return to Main Menu")
        print_100_equals()

        choice = input("\nEnter your choice: ")
        if choice == '1':
            rp_summary_table()
            pause_for_input()
        elif choice == '2':
            rp_live_demo()
            pause_for_input()
        elif choice == '3':
            rp_vs_baseline()
            pause_for_input()
        elif choice == '0':
            return
        else:
            print("Invalid choice, please try again.")
            pause_for_input()


def rp_summary_table():
    clear_screen()
    print_title()
    print("Random Projection Summary Table: Practice Runs (broken down by dimension)")
    print_100_equals()

    rp_kmeans_summary = pd.read_csv(KMEANS_METRICS_PATH_PRACTICE).query("Method == 'Random Projection'")
    rp_knn_summary    = pd.read_csv(KNN_METRICS_PATH_PRACTICE).query("Method == 'Random Projection'")

    # Same grouping logic as the PCA summary so the two tables stay
    # row-aligned and easy to compare side-by-side.
    kmeans_by_d = rp_kmeans_summary.groupby(['Dataset', 'Dimensions']).agg(
        ARI=('Adjusted Rand Index (ARI)', 'mean'),
        NMI=('Normalized Mutual Information (NMI)', 'mean'),
        Silhouette=('Silhouette Score (SS)', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
    ).round(4).reset_index()

    knn_by_d = rp_knn_summary.groupby(['Dataset', 'Dimensions']).agg(
        Accuracy=('Accuracy', 'mean'),
        F1=('F1 Score', 'mean'),
        AUC=('ROC AUC Score', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
    ).round(4).reset_index()

    for dataset in ['Small', 'Medium', 'Large']:
        print(f"\n[{dataset}] Random Projection + k-Means (one row per dimension)")
        rows = kmeans_by_d.query("Dataset == @dataset").sort_values('Dimensions')
        if rows.empty:
            print("  (no Random Projection k-Means runs logged for this dataset)")
        else:
            print(rows.drop(columns=['Dataset']).to_string(index=False))

        print(f"\n[{dataset}] Random Projection + kNN (one row per dimension)")
        rows = knn_by_d.query("Dataset == @dataset").sort_values('Dimensions')
        if rows.empty:
            print("  (no Random Projection kNN runs logged for this dataset)")
        else:
            print(rows.drop(columns=['Dataset']).to_string(index=False))

        print_100_equals()


def rp_live_demo():
    clear_screen()
    print_title()
    print("Random Projection Live Demo: run RP now and evaluate it immediately")
    print_100_equals()

    dataset_options = {
        '1': {
            'name': 'Small',
            'description': 'High-Dimensionality Dataset — Kaggle',
            'dimensions': x_small_dataset_scaled.shape[1],
            'classes': 2,
            'multiclass': False,
            'kmeans_X': x_small_dataset_scaled,
            'kmeans_y': y_small_dataset,
            'ss_sample_size': None,
        },
        '2': {
            'name': 'Medium',
            'description': 'UCI HAR Dataset',
            'dimensions': X_train_medium_dataset.shape[1],
            'classes': 6,
            'multiclass': True,
            'kmeans_X': X_train_medium_dataset,
            'kmeans_y': y_train_medium_dataset,
            'knn_train_X': X_train_medium_dataset,
            'knn_train_y': y_train_medium_dataset,
            'knn_test_X': X_test_medium_dataset,
            'knn_test_y': y_test_medium_dataset,
            'ss_sample_size': None,
        },
        '3': {
            'name': 'Large',
            'description': 'Fashion-MNIST',
            'dimensions': x_large_dataset_train_scaled.shape[1],
            'classes': 10,
            'multiclass': True,
            'kmeans_X': x_large_dataset_train_scaled,
            'kmeans_y': y_large_dataset_train,
            'knn_train_X': x_large_dataset_train_scaled,
            'knn_train_y': y_large_dataset_train,
            'knn_test_X': x_large_dataset_test_scaled,
            'knn_test_y': y_large_dataset_test,
            'ss_sample_size': 1000,
        },
    }

    print("Choose a dataset:")
    print("1. Small Dataset")
    print("2. Medium Dataset")
    print("3. Large Dataset")
    dataset_choice = input("\nEnter your choice: ")

    if dataset_choice not in dataset_options:
        print("Invalid dataset choice.")
        return

    config = dataset_options[dataset_choice]
    dataset_name = config['name']

    print_100_equals()
    print("Choose an algorithm:")
    print("1. Random Projection + k-Means")
    print("2. Random Projection + kNN")
    print("3. Run both")
    algorithm_choice = input("\nEnter your choice: ")

    if algorithm_choice not in {'1', '2', '3'}:
        print("Invalid algorithm choice.")
        return

    knn_data = None
    max_dimensions = []

    if algorithm_choice in {'1', '3'}:
        max_dimensions.append(max_rp_dimension(config['kmeans_X']))

    if algorithm_choice in {'2', '3'}:
        if dataset_name == 'Small':
            knn_data = train_test_split(
                x_small_dataset_scaled,
                y_small_dataset,
                test_size=0.3,
                random_state=42,
                stratify=y_small_dataset,
            )
            max_dimensions.append(max_rp_dimension(knn_data[0]))
        else:
            max_dimensions.append(max_rp_dimension(config['knn_train_X']))

    max_d = min(max_dimensions)
    suggested_dimensions = [d for d in RP_DIMENSIONS if d <= max_d]

    print_100_equals()
    print(f"{dataset_name} has {config['kmeans_X'].shape[1]} original dimensions.")
    print(f"Suggested Random Projection dimensions: {suggested_dimensions}")
    dimension_choice = input(f"Enter Random Projection target dimension d (1 to {max_d}): ")

    try:
        d = int(dimension_choice)
    except ValueError:
        print("Dimension must be a whole number.")
        return

    if d < 1 or d > max_d:
        print(f"Invalid dimension. For {dataset_name}, choose a value from 1 to {max_d}.")
        return

    print_100_equals()
    print(f"Running Random Projection live demo for {dataset_name} with d={d}...")

    if algorithm_choice in {'1', '3'}:
        z_train, _, _rp, rp_time = apply_random_projection(
            config['kmeans_X'],
            d,
            random_state=42,
        )
        metrics = run_rp_kmeans(
            z_train,
            config['kmeans_y'],
            k=config['classes'],
            ss_sample_size=config['ss_sample_size'],
            random_state=42,
        )
        metrics['Time Taken (seconds)'] += rp_time

        log_kmeans_demo_result({
            'Dataset': dataset_name,
            'Method': 'Random Projection',
            'Dimensions': d,
            **metrics,
        })

        # Random Projection has no explained_variance_ratio_, so the table
        # is one column shorter than the PCA equivalent.
        kmeans_table = pd.DataFrame([{
            'Dataset': dataset_name,
            'Dimensions': d,
            'ARI': metrics['Adjusted Rand Index (ARI)'],
            'NMI': metrics['Normalized Mutual Information (NMI)'],
            'Silhouette': metrics['Silhouette Score (SS)'],
            'Time_s': metrics['Time Taken (seconds)'],
        }]).round(4)

        print("\nRandom Projection + k-Means Live Results:")
        print(kmeans_table.to_string(index=False))

    if algorithm_choice in {'2', '3'}:
        if knn_data is not None:
            X_train, X_test, y_train, y_test = knn_data
        else:
            X_train = config['knn_train_X']
            y_train = config['knn_train_y']
            X_test = config['knn_test_X']
            y_test = config['knn_test_y']

        z_train, z_test, _rp, rp_time = apply_random_projection(
            X_train,
            d,
            X_test=X_test,
            random_state=42,
        )
        metrics = run_rp_knn(
            z_train,
            y_train,
            z_test,
            y_test,
            n_neighbors=5,
            multiclass=config['multiclass'],
        )
        metrics['Time Taken (seconds)'] += rp_time

        log_knn_demo_result({
            'Dataset': dataset_name,
            'Method': 'Random Projection',
            'Dimensions': d,
            **metrics,
        })

        knn_table = pd.DataFrame([{
            'Dataset': dataset_name,
            'Dimensions': d,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1 Score'],
            'AUC': metrics['ROC AUC Score'],
            'Time_s': metrics['Time Taken (seconds)'],
        }]).round(4)

        print("\nRandom Projection + kNN Live Results:")
        print(knn_table.to_string(index=False))

    print_100_equals()
    print("Live demo complete. Results were logged to the demo-runs CSV files. Please check the results/metrics/demo-runs directory.")


def rp_vs_baseline():
    clear_screen()
    print_title()
    print("Random Projection vs. Baseline: how much do we gain (or lose) by reducing dimensions?")
    print_100_equals()

    kmeans_all = pd.read_csv(KMEANS_METRICS_PATH_PRACTICE).query("Method in ['Baseline', 'Random Projection']")
    knn_all    = pd.read_csv(KNN_METRICS_PATH_PRACTICE).query("Method in ['Baseline', 'Random Projection']")

    kmeans_agg = kmeans_all.groupby(['Dataset', 'Method', 'Dimensions']).agg(
        ARI=('Adjusted Rand Index (ARI)', 'mean'),
        NMI=('Normalized Mutual Information (NMI)', 'mean'),
        Silhouette=('Silhouette Score (SS)', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
        n_runs=('Time Taken (seconds)', 'count'),
    ).reset_index()

    knn_agg = knn_all.groupby(['Dataset', 'Method', 'Dimensions']).agg(
        Accuracy=('Accuracy', 'mean'),
        F1=('F1 Score', 'mean'),
        AUC=('ROC AUC Score', 'mean'),
        Time_s=('Time Taken (seconds)', 'mean'),
        n_runs=('Time Taken (seconds)', 'count'),
    ).reset_index()

    for dataset in ['Small', 'Medium', 'Large']:
        _print_method_vs_baseline_for_algo(
            dataset,
            kmeans_agg.query("Dataset == @dataset"),
            algorithm_label='Random Projection + k-Means',
            quality_col='ARI',
            display_cols=['Method', 'Dimensions', 'ARI', 'NMI', 'Silhouette', 'Time_s'],
        )

        _print_method_vs_baseline_for_algo(
            dataset,
            knn_agg.query("Dataset == @dataset"),
            algorithm_label='Random Projection + kNN',
            quality_col='Accuracy',
            display_cols=['Method', 'Dimensions', 'Accuracy', 'F1', 'AUC', 'Time_s'],
        )

        print_100_equals()

def comparison_of_dimensionality_reduction_results():
    while True:
        clear_screen()
        print_title()
        print("Comparison: PCA vs. Random Projection (head-to-head)")

        print_100_equals()
        print("Please select option:")
        print("1. Practice Runs Only")
        print("2. Demo Runs Only")
        print("3. Combined View (Practice + Demo)")
        print("0. Return to Main Menu")
        print_100_equals()

        choice = input("\nEnter your choice: ")
        if choice == '1':
            _pca_vs_rp_view(source='practice')
            pause_for_input()
        elif choice == '2':
            _pca_vs_rp_view(source='demo')
            pause_for_input()
        elif choice == '3':
            _pca_vs_rp_view(source='combined')
            pause_for_input()
        elif choice == '0':
            return
        else:
            print("Invalid choice, please try again.")
            pause_for_input()


def _pca_vs_rp_view(source):
    clear_screen()
    print_title()
    source_label = {
        'practice': 'Practice Runs',
        'demo': 'Demo Runs',
        'combined': 'Combined (Practice + Demo)',
    }[source]
    print(f"PCA vs. Random Projection — {source_label}")
    print_100_equals()

    kmeans_df = _load_dr_metrics(
        source,
        practice_path=KMEANS_METRICS_PATH_PRACTICE,
        demo_path=KMEANS_METRICS_PATH_LIVE_DEMO,
    ).query("Method in ['PCA', 'Random Projection']")

    knn_df = _load_dr_metrics(
        source,
        practice_path=KNN_METRICS_PATH_PRACTICE,
        demo_path=KNN_METRICS_PATH_LIVE_DEMO,
    ).query("Method in ['PCA', 'Random Projection']")

    if kmeans_df.empty and knn_df.empty:
        print(f"\n  (no {source_label} data logged yet)")
        if source in {'demo', 'combined'}:
            print("  Tip: run the PCA or Random Projection live demos to populate the demo-runs CSV.")
        return

    kmeans_agg_map = {
        'ARI':        ('Adjusted Rand Index (ARI)', 'mean'),
        'NMI':        ('Normalized Mutual Information (NMI)', 'mean'),
        'Silhouette': ('Silhouette Score (SS)', 'mean'),
        'Time_s':     ('Time Taken (seconds)', 'mean'),
    }

    knn_agg_map = {
        'Accuracy': ('Accuracy', 'mean'),
        'F1':       ('F1 Score', 'mean'),
        'AUC':      ('ROC AUC Score', 'mean'),
        'Time_s':   ('Time Taken (seconds)', 'mean'),
    }

    for dataset in ['Small', 'Medium', 'Large']:
        _print_pca_vs_rp_pivot(
            dataset=dataset,
            rows=kmeans_df.query("Dataset == @dataset"),
            algorithm_label='k-Means',
            agg_map=kmeans_agg_map,
        )

        _print_pca_vs_rp_pivot(
            dataset=dataset,
            rows=knn_df.query("Dataset == @dataset"),
            algorithm_label='kNN',
            agg_map=knn_agg_map,
        )

        print_100_equals()


def _load_dr_metrics(source, practice_path, demo_path):
    frames = []
    if source in {'practice', 'combined'} and practice_path.exists():
        frames.append(pd.read_csv(practice_path))
    if source in {'demo', 'combined'} and demo_path.exists():
        frames.append(pd.read_csv(demo_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _print_pca_vs_rp_pivot(dataset, rows, algorithm_label, agg_map):
    print(f"\n[{dataset}] {algorithm_label}: PCA vs. Random Projection")

    if rows.empty:
        print("  (no PCA or Random Projection runs logged for this dataset)")
        return

    metric_cols = list(agg_map.keys())

    # Average across repeated runs at the same (Method, Dimensions).
    agg = rows.groupby(['Method', 'Dimensions']).agg(**agg_map).reset_index()

    # Pivot so each metric becomes two columns (one per method).
    pivot = agg.pivot(index='Dimensions', columns='Method', values=metric_cols)

    # Interleave columns so each metric's PCA and RP pair sit next to each
    # other: [PCA_ARI, RP_ARI, PCA_NMI, RP_NMI, ...]
    ordered_cols = []
    for metric in metric_cols:
        for method in ['PCA', 'Random Projection']:
            if (metric, method) in pivot.columns:
                ordered_cols.append((metric, method))
    pivot = pivot[ordered_cols]

    # Flatten the MultiIndex column names into something readable.
    method_short = {'PCA': 'PCA', 'Random Projection': 'RP'}
    pivot.columns = [f"{method_short[method]}_{metric}" for metric, method in pivot.columns]

    print(pivot.round(4).to_string())

if __name__ == "__main__":
    while True:
        choice = user_menu()
        if choice == '0':
            print("\nExiting program... thank you for using the Dimensionality Reduction Program!")
            break
        dispatch_choice(choice)
        pause_for_input()
    