from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from joblib import dump
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "processed"
DATASETS_DIR = DATA_ROOT / "datasets"
MODELS_DIR = DATA_ROOT / "models"
RESULTS_DIR = DATA_ROOT / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def select_feature_columns(df: pd.DataFrame, target: str) -> list:
    drop_cols = {
        "app_id", "release_id", "date",
        "review_count", "average_rating",
        "target_review_count", "target_average_rating",
        "target_review_bucket", "target_rating_bucket",
        target,
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in drop_cols]


def fit_models(X: pd.DataFrame, y: pd.Series, target_name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    rf.fit(X, y)
    dump(rf, MODELS_DIR / f"rf_{target_name}.joblib")
    logger.info(f"Saved RF model -> {MODELS_DIR / f'rf_{target_name}.joblib'}")

    xgb_base = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
    )
    param_grid = {
        "n_estimators": [400, 800],
        "learning_rate": [0.03, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [1.0, 5.0],
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X, y)
    xgb_best = gs.best_estimator_
    dump(xgb_best, MODELS_DIR / f"xgb_{target_name}.joblib")
    logger.info(f"Saved XGB model -> {MODELS_DIR / f'xgb_{target_name}.joblib'}")

    meta = {
        "target": target_name,
        "feature_columns": list(X.columns),
        "n_train_rows": int(X.shape[0]),
        "xgb_best_params": gs.best_params_,
        "xgb_cv_score_neg_rmse": float(gs.best_score_),
        "rf_params": rf.get_params(),
    }
    meta_path = MODELS_DIR / f"meta_{target_name}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata -> {meta_path}")


def main():
    train_path = DATASETS_DIR / "train_dataset.csv"
    df = pd.read_csv(train_path, parse_dates=["date"])
    targets = [t for t in ["target_review_count", "target_average_rating"] if t in df.columns]
    for target in targets:
        logger.info(f"=== Training for target: {target} ===")
        feat_cols = select_feature_columns(df, target)
        X = df[feat_cols].fillna(0)
        y = df[target].astype(float)
        fit_models(X, y, target)
    print("\nTraining complete. Models stored in:", MODELS_DIR)


if __name__ == "__main__":
    main()
