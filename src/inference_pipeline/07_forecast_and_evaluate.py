import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, median_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "processed"
DATASETS_DIR = DATA_ROOT / "datasets"
MODELS_DIR = DATA_ROOT / "models"
RESULTS_DIR = DATA_ROOT / "results"
PLOTS_DIR = DATA_ROOT / "plots"

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(np.maximum(0, y_true)),
                                      np.log1p(np.maximum(0, y_pred))))


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)
    return 100.0 * np.mean(diff)


def compute_naive_baselines(df_train, df_test, target_col):
    baselines = {}

    # Group by app for app-specific baselines
    train_grouped = df_train.groupby('app_id')
    test_grouped = df_test.groupby('app_id')

    predictions = []

    for app_id in df_test['app_id'].unique():
        test_app = test_grouped.get_group(app_id) if app_id in test_grouped.groups else df_test[
            df_test['app_id'] == app_id]

        if app_id in train_grouped.groups:
            train_app = train_grouped.get_group(app_id)
            train_values = train_app[target_col].values

            # Last value baseline
            if len(train_values) > 0:
                last_value = train_values[-1]
            else:
                last_value = df_train[target_col].median()

            # Moving average baseline (MA3)
            if len(train_values) >= 3:
                ma3_value = np.mean(train_values[-3:])
            elif len(train_values) > 0:
                ma3_value = np.mean(train_values)
            else:
                ma3_value = df_train[target_col].median()

            # Global median baseline
            global_median = df_train[target_col].median()

        else:
            # If app not in training, use global statistics
            last_value = df_train[target_col].median()
            ma3_value = df_train[target_col].median()
            global_median = df_train[target_col].median()

        for _, row in test_app.iterrows():
            predictions.append({
                'app_id': app_id,
                'release_id': row['release_id'],
                'actual': row[target_col],
                'last_value': last_value,
                'ma3': ma3_value,
                'global_median': global_median
            })

    pred_df = pd.DataFrame(predictions)

    # Compute metrics for each baseline
    baselines['last_value'] = compute_regression_metrics(pred_df['actual'], pred_df['last_value'])
    baselines['ma3'] = compute_regression_metrics(pred_df['actual'], pred_df['ma3'])
    baselines['global_median'] = compute_regression_metrics(pred_df['actual'], pred_df['global_median'])

    return baselines, pred_df


def compute_regression_metrics(y_true, y_pred):
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=0.0)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {metric: np.nan for metric in ['mae', 'rmse', 'rmsle', 'smape', 'mape', 'mdae', 'r2', 'spearman']}

    metrics = {}

    try:
        metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
    except:
        metrics['mae'] = np.nan

    try:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    except:
        metrics['rmse'] = np.nan

    try:
        metrics['rmsle'] = rmsle(y_true_clean, y_pred_clean)
    except:
        metrics['rmsle'] = np.nan

    try:
        metrics['smape'] = smape(y_true_clean, y_pred_clean)
    except:
        metrics['smape'] = np.nan

    try:
        nonzero_mask = y_true_clean != 0
        if np.sum(nonzero_mask) > 0:
            metrics['mape'] = mean_absolute_percentage_error(
                y_true_clean[nonzero_mask], y_pred_clean[nonzero_mask]
            ) * 100
        else:
            metrics['mape'] = np.nan
    except:
        metrics['mape'] = np.nan

    try:
        metrics['mdae'] = median_absolute_error(y_true_clean, y_pred_clean)
    except:
        metrics['mdae'] = np.nan

    try:
        metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
    except:
        metrics['r2'] = np.nan

    try:
        correlation, _ = spearmanr(y_true_clean, y_pred_clean)
        metrics['spearman'] = correlation
    except:
        metrics['spearman'] = np.nan

    return metrics


def load_model_and_predict(model_path, X_test, metadata):
    try:
        model = load(model_path)

        if 'feature_columns' in metadata:
            feature_cols = metadata['feature_columns']
            X_test_selected = X_test[feature_cols].fillna(0)
        else:
            X_test_selected = X_test.fillna(0)

        predictions = model.predict(X_test_selected)

        if 'xgb_log1p_postprocess' in metadata and metadata['xgb_log1p_postprocess'] == 'expm1':
            predictions = np.expm1(predictions)

        if 'rating_clip' in metadata:
            clip_min, clip_max = metadata['rating_clip']
            predictions = np.clip(predictions, clip_min, clip_max)

        return predictions
    except Exception as e:
        logger.error(f"Error loading/predicting with {model_path}: {e}")
        return None


def create_prediction_plots(y_true, y_pred, model_name, target_name, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'{model_name}: Predicted vs Actual\n{target_name}')
    ax1.grid(True, alpha=0.3)

    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{model_name}: Residual Plot\n{target_name}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_name = f"{model_name.lower().replace(' ', '_')}_{target_name}_plots.png"
    plt.savefig(save_dir / plot_name, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_name


def analyze_worst_predictions(df_pred, target_col, model_name, k=10):
    df_pred = df_pred.copy()
    df_pred['abs_error'] = np.abs(df_pred[f'{model_name}_pred'] - df_pred[target_col])
    df_pred['rel_error'] = df_pred['abs_error'] / (np.abs(df_pred[target_col]) + 1e-8)

    worst_abs = df_pred.nlargest(k, 'abs_error')[
        ['app_id', 'release_id', target_col, f'{model_name}_pred', 'abs_error', 'rel_error']
    ]
    worst_rel = df_pred.nlargest(k, 'rel_error')[
        ['app_id', 'release_id', target_col, f'{model_name}_pred', 'abs_error', 'rel_error']
    ]

    return {
        'worst_absolute': worst_abs,
        'worst_relative': worst_rel
    }


def main():
    logger.info("Starting forecasting and evaluation...")

    # Load datasets
    train_path = DATASETS_DIR / "train_dataset.csv"
    test_path = DATASETS_DIR / "test_dataset.csv"

    logger.info(f"Loading train dataset: {train_path}")
    df_train = pd.read_csv(train_path, parse_dates=["date"])

    logger.info(f"Loading test dataset: {test_path}")
    df_test = pd.read_csv(test_path, parse_dates=["date"])

    logger.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Target columns to evaluate
    targets = [col for col in ["target_review_count", "target_average_rating"]
               if col in df_test.columns]

    logger.info(f"Found targets: {targets}")

    # Results storage
    all_results = {}
    run_manifest = {
        'timestamp': datetime.now().isoformat(),
        'train_size': len(df_train),
        'test_size': len(df_test),
        'targets': targets,
        'artifacts': {}
    }

    for target in targets:
        logger.info(f"\n=== Evaluating target: {target} ===")

        target_results = {}

        drop_cols = {
            "app_id", "release_id", "date",
            "review_count", "average_rating",
            "target_review_count", "target_average_rating",
            "target_review_bucket", "target_rating_bucket"
        }
        feature_cols = [c for c in df_test.select_dtypes(include=[np.number]).columns
                        if c not in drop_cols]

        X_test = df_test[feature_cols]
        y_test = df_test[target]

        logger.info("Computing naive baselines...")
        baselines, baseline_df = compute_naive_baselines(df_train, df_test, target)
        target_results['baselines'] = baselines

        baseline_path = RESULTS_DIR / f"baseline_predictions_{target}.csv"
        baseline_df.to_csv(baseline_path, index=False)
        run_manifest['artifacts'][f'baseline_predictions_{target}'] = str(baseline_path)

        model_files = {
            'random_forest': MODELS_DIR / f"rf_{target}.joblib",
            'xgboost': MODELS_DIR / f"xgb_{target}.joblib",
        }

        if target == "target_review_count":
            model_files['xgboost_poisson'] = MODELS_DIR / f"xgb_poisson_{target}.joblib"
            model_files['xgboost_log1p'] = MODELS_DIR / f"xgb_log1p_{target}.joblib"
        elif target == "target_average_rating":
            model_files['huber_regressor'] = MODELS_DIR / f"huber_{target}.joblib"

        meta_path = MODELS_DIR / f"meta_{target}.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        model_predictions = {'actual': y_test}

        for model_name, model_path in model_files.items():
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                continue

            logger.info(f"Evaluating {model_name}...")

            predictions = load_model_and_predict(model_path, X_test, metadata)

            if predictions is not None:
                metrics = compute_regression_metrics(y_test, predictions)
                target_results[model_name] = metrics

                # Store predictions for later analysis
                model_predictions[f'{model_name}_pred'] = predictions

                # Create plots
                plot_file = create_prediction_plots(
                    y_test, predictions, model_name, target, PLOTS_DIR
                )
                run_manifest['artifacts'][f'{model_name}_{target}_plot'] = plot_file

                logger.info(f"{model_name} - MAE: {metrics['mae']:.4f}, "
                            f"RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            else:
                logger.error(f"Failed to get predictions from {model_name}")

        # Save all predictions to CSV
        pred_df = pd.DataFrame(model_predictions)
        pred_df['app_id'] = df_test['app_id'].values
        pred_df['release_id'] = df_test['release_id'].values
        pred_df['date'] = df_test['date'].values

        pred_path = RESULTS_DIR / f"model_predictions_{target}.csv"
        pred_df.to_csv(pred_path, index=False)
        run_manifest['artifacts'][f'model_predictions_{target}'] = str(pred_path)

        # Analyze worst predictions for best model
        if len(target_results) > 1:  # More than just baselines
            model_metrics = {k: v for k, v in target_results.items() if k != 'baselines'}
            if model_metrics:
                best_model = min(model_metrics.keys(),
                                 key=lambda x: model_metrics[x]['rmse'])

                worst_cases = analyze_worst_predictions(
                    pred_df, 'actual', best_model, k=10
                )

                # Save worst cases
                worst_path = RESULTS_DIR / f"worst_predictions_{target}_{best_model}.json"
                with open(worst_path, 'w') as f:
                    json.dump({
                        'worst_absolute': worst_cases['worst_absolute'].to_dict('records'),
                        'worst_relative': worst_cases['worst_relative'].to_dict('records')
                    }, f, indent=2)
                run_manifest['artifacts'][f'worst_cases_{target}'] = str(worst_path)

        # Save target-specific results
        results_path = RESULTS_DIR / f"metrics_{target}.json"
        with open(results_path, 'w') as f:
            json.dump(target_results, f, indent=2, default=str)
        run_manifest['artifacts'][f'metrics_{target}'] = str(results_path)

        all_results[target] = target_results

    # Create summary comparison plots
    logger.info("Creating summary comparison plots...")

    for target in targets:
        if target not in all_results:
            continue

        target_results = all_results[target]

        models = []
        rmse_values = []
        mae_values = []

        for baseline_name, metrics in target_results.get('baselines', {}).items():
            models.append(f"Baseline: {baseline_name}")
            rmse_values.append(metrics['rmse'])
            mae_values.append(metrics['mae'])

        for model_name, metrics in target_results.items():
            if model_name != 'baselines':
                models.append(model_name.replace('_', ' ').title())
                rmse_values.append(metrics['rmse'])
                mae_values.append(metrics['mae'])

        if len(models) > 0:
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # RMSE comparison
            bars1 = ax1.bar(models, rmse_values)
            ax1.set_title(f'RMSE Comparison - {target}')
            ax1.set_ylabel('RMSE')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # MAE comparison
            bars2 = ax2.bar(models, mae_values)
            ax2.set_title(f'MAE Comparison - {target}')
            ax2.set_ylabel('MAE')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            comparison_plot = f"model_comparison_{target}.png"
            plt.savefig(PLOTS_DIR / comparison_plot, dpi=150, bbox_inches='tight')
            plt.close()

            run_manifest['artifacts'][f'comparison_plot_{target}'] = comparison_plot

    # Save complete results and manifest
    logger.info("Saving final results...")

    complete_results_path = RESULTS_DIR / "complete_evaluation_results.json"
    with open(complete_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    manifest_path = RESULTS_DIR / "run_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(run_manifest, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    for target in targets:
        if target not in all_results:
            continue

        logger.info(f"\nTarget: {target}")
        logger.info("-" * 40)

        target_results = all_results[target]

        # Print baselines
        logger.info("Baselines:")
        for baseline_name, metrics in target_results.get('baselines', {}).items():
            logger.info(f"  {baseline_name:15s}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

        # Print ML models
        logger.info("ML Models:")
        for model_name, metrics in target_results.items():
            if model_name != 'baselines':
                logger.info(
                    f"  {model_name:15s}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

    logger.info(f"\nAll results saved to: {RESULTS_DIR}")
    logger.info(f"All plots saved to: {PLOTS_DIR}")
    logger.info(f"Run manifest: {manifest_path}")

    return all_results, run_manifest


if __name__ == "__main__":
    results, manifest = main()
    print("\nForecasting and evaluation completed successfully!")