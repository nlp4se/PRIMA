import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

from src.utils.config import DATASETS_DIR, LOG_LEVEL, LOG_FORMAT


def compute_correlations(df: pd.DataFrame, target_col: str):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
        [target_col, 'review_count', 'average_rating', 'target_review_count', 'target_average_rating'], errors='ignore'
    )
    pearson_scores = {}
    for col in numeric_cols:
        try:
            corr, _ = pearsonr(df[col], df[target_col])
            pearson_scores[col] = corr
        except Exception:
            continue
    return pearson_scores


def compute_mutual_info(df: pd.DataFrame, target_col: str):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
        [target_col, 'review_count', 'average_rating', 'target_review_count', 'target_average_rating'], errors='ignore'
    )
    X = df[numeric_cols].fillna(0)
    y = df[target_col].fillna(0)
    scores = mutual_info_regression(X, y, random_state=42)
    return dict(zip(numeric_cols, scores))


def display_top(scores: dict, label: str, k=10):
    print(f"\n=== Top {k} features by {label} ===")
    top = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:k]
    for feat, score in top:
        print(f"{feat}: {score:.4f}")


def main():
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)

    train_path = Path(DATASETS_DIR) / "train_dataset.csv"
    df = pd.read_csv(train_path)

    print("\n=== Evaluating Feature–Target Correlation ===")
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    for target in ['target_review_count', 'target_average_rating']:
        if target not in df.columns:
            continue
        print(f"\n>>> Analysis for: {target}")
        pearson = compute_correlations(df, target)
        mi = compute_mutual_info(df, target)
        display_top(pearson, f"Pearson correlation with {target}")
        display_top(mi, f"Mutual Information with {target}")


if __name__ == '__main__':
    main()
