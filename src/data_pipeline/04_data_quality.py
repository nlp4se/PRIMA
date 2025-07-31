import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr

from src.utils.config import DATASETS_DIR, LOG_LEVEL, LOG_FORMAT


def main():
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)

    # Paths
    train_path = Path(DATASETS_DIR) / "train_dataset.csv"
    test_path = Path(DATASETS_DIR) / "test_dataset.csv"
    plots_dir = Path(DATASETS_DIR) / "plots_quality"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading train: {train_path}")
    train = pd.read_csv(train_path, parse_dates=["date"] )
    logging.info(f"Loading test : {test_path}")
    test  = pd.read_csv(test_path,  parse_dates=["date"] )

    # Basic summary
    print("\n=== Dataset Sizes ===")
    print(f"Train: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"Test : {test.shape[0]} rows, {test.shape[1]} columns")

    # 1) Kolmogorov-Smirnov test on target distributions
    ks_stat, ks_p = ks_2samp(
        train['target_review_count'],
        test['target_review_count']
    )
    print("\n=== Target Distribution Shift ===")
    print(f"KS statistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")

    # 2) Top feature correlations in train
    num_cols = train.select_dtypes(include=[np.number]).columns.drop(['target_review_count', 'review_count'])
    corrs = {}
    for col in num_cols:
        try:
            corr, _ = pearsonr(train[col], train['target_review_count'])
            corrs[col] = corr
        except ValueError:
            continue
    top_feats = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("\n=== Top 10 Features by Pearson Correlation with Target ===")
    for feat, corr in top_feats:
        print(f"{feat}: {corr:.3f}")

    # 3) Plot target histograms
    plt.figure()
    train['target_review_count'].hist(bins=30)
    plt.title('Train Target Distribution')
    plt.xlabel('target_review_count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(plots_dir / 'train_target_dist.png')
    plt.close()

    plt.figure()
    test['target_review_count'].hist(bins=30)
    plt.title('Test Target Distribution')
    plt.xlabel('target_review_count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(plots_dir / 'test_target_dist.png')
    plt.close()

    # 4) Plot release years
    plt.figure()
    train['date'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Train Releases by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(plots_dir / 'train_years.png')
    plt.close()

    plt.figure()
    test['date'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Test Releases by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(plots_dir / 'test_years.png')
    plt.close()

    print(f"\nPlots and stats saved to: {plots_dir}")


if __name__ == '__main__':
    main()
