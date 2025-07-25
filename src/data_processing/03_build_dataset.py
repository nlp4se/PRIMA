import logging
import pandas as pd

from src.utils.config import *


def main():
    # Initialize logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )

    # Define file paths
    features_file = Path(FEATURES_DIR) / "extracted_features.csv"
    train_file = Path(DATASETS_DIR) / "train_dataset.csv"
    test_file = Path(DATASETS_DIR) / "test_dataset.csv"

    logging.info(f"Loading features from {features_file}")
    df = pd.read_csv(features_file, parse_dates=["date"])

    # Sort releases by app and date
    df.sort_values(["app_id", "date"], inplace=True)

    # Create target: number of reviews for next release
    df['target_review_count'] = (
        df.groupby('app_id')['review_count'].shift(-1)
    )

    # Drop entries without a next-release target
    df_model = df.dropna(subset=['target_review_count']).copy()
    df_model['target_review_count'] = df_model['target_review_count'].astype(int)

    # Prepare train/test containers
    train_list = []
    test_list = []

    # Per-app chronological split
    for app, group in df_model.groupby('app_id'):
        n = len(group)
        cutoff = int(n * TEMPORAL_SPLIT_RATIO)
        if cutoff < 1:
            # Too few releases: assign all to train
            train_list.append(group)
        else:
            train_list.append(group.iloc[:cutoff])
            test_list.append(group.iloc[cutoff:])

    # Combine
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # Ensure output directory
    Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)

    # Save
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    logging.info(f"Per-app split complete: {len(train_df)} train rows, {len(test_df)} test rows")
    print("\nBuild datasets complete with per-app split!")
    print(f"  Train rows: {len(train_df)} -> {train_file}")
    print(f"  Test  rows: {len(test_df)} -> {test_file}")


if __name__ == '__main__':
    main()
