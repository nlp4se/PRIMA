from pathlib import Path
import json
import logging
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import HuberRegressor
from joblib import dump
from xgboost import XGBClassifier
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "processed"
DATASETS_DIR = DATA_ROOT / "datasets"
MODELS_DIR = DATA_ROOT / "models"
RESULTS_DIR = DATA_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def select_feature_columns(df: pd.DataFrame, target: str) -> list:
    """Enhanced feature selection with outlier handling"""
    drop_cols = {
        "app_id", "release_id", "date",
        "review_count", "average_rating",
        "target_review_count", "target_average_rating",
        "target_review_bucket", "target_rating_bucket",
        target,
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    # Remove features with extreme outliers
    X_features = df[feature_cols].fillna(0)

    # Remove columns with >95% zeros or constant values
    non_zero_ratio = (X_features != 0).mean()
    feature_cols = [c for c in feature_cols if non_zero_ratio[c] > 0.05]

    # Cap extreme values at 99th percentile
    for col in feature_cols:
        if col in X_features.columns:
            cap_value = X_features[col].quantile(0.99)
            if cap_value > 0:
                X_features[col] = np.clip(X_features[col], 0, cap_value)

    return feature_cols


def grid_iter(param_grid: dict):
    """Generate parameter combinations"""
    keys = sorted(param_grid)
    for values in itertools.product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, values))


def to_cv_params(cfg: dict, objective: str):
    p = {
        "objective": objective,
        "tree_method": "hist",
        "max_depth": cfg["max_depth"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "reg_lambda": cfg["reg_lambda"],
        "reg_alpha": cfg.get("reg_alpha", 1.0),
        "gamma": cfg.get("gamma", 0.1),
        "min_child_weight": cfg.get("min_child_weight", 10),
        "max_delta_step": cfg.get("max_delta_step", 1),
        "eta": cfg["learning_rate"],
        "seed": 42,
        "nthread": -1,
    }
    return p


def xgb_cv_select(X, y, objective, metric, base, grid, nfold=3, num_boost_round=500, es_rounds=30):
    base_fixed = {
        **base,
        "reg_alpha": 1.0,  # L1 regularization
        "reg_lambda": 10.0,  # L2 regularization
        "gamma": 0.1,  # Minimum split loss
        "min_child_weight": 10,  # Minimum samples in leaf
        "max_delta_step": 1,  # Maximum delta step for extreme values
    }

    if "count" in objective or "poisson" in objective:
        y_capped = np.clip(y, 0, 1000)  # Cap at 1000 reviews
    else:
        y_capped = np.clip(y, 1.0, 5.0) if "reg:" in objective else y

    dtrain = xgb.DMatrix(X.fillna(0), label=y_capped)

    best_score = float('inf') if 'rmse' in metric else -float('inf')
    best_params = None
    best_round = 0

    for params in grid_iter(grid):
        cv_params = to_cv_params({**base_fixed, **params}, objective)

        try:
            cv_result = xgb.cv(
                cv_params,
                dtrain,
                num_boost_round=num_boost_round,
                nfold=nfold,
                metrics=[metric],
                early_stopping_rounds=es_rounds,
                seed=42,
                verbose_eval=False,
                show_stdv=False
            )

            if 'rmse' in metric:
                score = cv_result[f'test-{metric}-mean'].min()
                is_better = score < best_score
            else:
                score = cv_result[f'test-{metric}-mean'].max()
                is_better = score > best_score

            if is_better:
                best_score = score
                best_params = {**base_fixed, **params}
                best_round = len(cv_result)

        except Exception as e:
            logger.warning(f"CV failed for params {params}: {e}")
            continue

    if best_params is None:
        best_params = base_fixed
        best_round = 100
        best_score = 999.0

    try:
        final_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_round,
            verbose_eval=False
        )
    except Exception as e:
        logger.error(f"Final model training failed: {e}")
        final_model = xgb.train(
            base_fixed,
            dtrain,
            num_boost_round=50,
            verbose_eval=False
        )

    return final_model, {
        "params": best_params,
        "best_round": best_round,
        "score": float(best_score)
    }


def train_classification_models(X, y_bucket, target_name):
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X.fillna(0), y_bucket)
    dump(rf_clf, MODELS_DIR / f"rf_clf_{target_name}.joblib")

    xgb_clf = XGBClassifier(
        objective='multi:softprob',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X.fillna(0), y_bucket)
    dump(xgb_clf, MODELS_DIR / f"xgb_clf_{target_name}.joblib")

    logger.info(f"Saved classification models for {target_name}")


def fit_models(X, y, target_name):

    logger.info(f"Training models for {target_name} with {X.shape[0]} samples, {X.shape[1]} features")
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_cleaned = X.copy()
    for col in X.columns:
        X_cleaned[col] = np.clip(X_cleaned[col], lower_bound[col], upper_bound[col])

    # 2. Target transformation
    if target_name == "target_review_count":
        y_transformed = np.log1p(np.maximum(0, y))
        use_log_transform = True
    else:
        y_transformed = np.clip(y, 1.0, 5.0)
        use_log_transform = False

    # 3. Feature scaling
    try:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_cleaned.fillna(0))
        X_processed = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    except Exception as e:
        logger.warning(f"Scaling failed: {e}, using original features")
        X_processed = X_cleaned.fillna(0)

    # Random Forest with better parameters
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_processed, y_transformed)
    dump(rf, MODELS_DIR / f"rf_{target_name}.joblib")
    logger.info(f"Saved RF model -> {MODELS_DIR / f'rf_{target_name}.joblib'}")

    # XGBoost training
    base = dict(
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        reg_alpha=1.0,
        gamma=0.1,
        min_child_weight=10,
    )

    grid = {
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [3, 4, 5],
        "reg_lambda": [5.0, 10.0, 20.0],
    }

    xgb_model, best_result = xgb_cv_select(
        X_processed, y_transformed,
        objective="reg:squarederror",
        metric="rmse",
        base=base, grid=grid,
        nfold=3, num_boost_round=500, es_rounds=30
    )
    dump(xgb_model, MODELS_DIR / f"xgb_{target_name}.joblib")
    logger.info(f"Saved XGB model -> {MODELS_DIR / f'xgb_{target_name}.joblib'}")

    extra = {
        "preprocessing": {
            "outlier_bounds": {col: [float(lower_bound[col]), float(upper_bound[col])] for col in X.columns},
            "use_log_transform": use_log_transform,
            "feature_pipeline": "RobustScaler"
        },
        "xgb_best_params": best_result["params"],
        "xgb_cv_best_round": int(best_result["best_round"]),
        "xgb_cv_best_score": float(best_result["score"]),
    }

    if target_name == "target_review_count":

        # XGBoost Poisson
        xgb_pois, best_pois = xgb_cv_select(
            X_processed, np.maximum(1, y),  # Poisson needs positive values
            objective="count:poisson",
            metric="poisson-nloglik",
            base=base, grid=grid,
            nfold=3, num_boost_round=500, es_rounds=30
        )
        dump(xgb_pois, MODELS_DIR / f"xgb_poisson_{target_name}.joblib")
        logger.info(f"Saved XGB Poisson model -> {MODELS_DIR / f'xgb_poisson_{target_name}.joblib'}")

        # XGBoost log1p
        y_log = np.log1p(np.maximum(0, y))
        xgb_log, best_log = xgb_cv_select(
            X_processed, y_log,
            objective="reg:squarederror",
            metric="rmse",
            base=base, grid=grid,
            nfold=3, num_boost_round=500, es_rounds=30
        )
        dump(xgb_log, MODELS_DIR / f"xgb_log1p_{target_name}.joblib")
        logger.info(f"Saved XGB log1p model -> {MODELS_DIR / f'xgb_log1p_{target_name}.joblib'}")

        extra.update({
            "xgb_poisson_params": best_pois["params"],
            "xgb_poisson_cv_best_round": int(best_pois["best_round"]),
            "xgb_poisson_cv_best_score": float(best_pois["score"]),
            "xgb_log1p_params": best_log["params"],
            "xgb_log1p_cv_best_round": int(best_log["best_round"]),
            "xgb_log1p_cv_best_score": float(best_log["score"]),
            "xgb_log1p_postprocess": "expm1",
        })

        # Classification models
        try:
            y_bucket = pd.qcut(y, q=4, labels=["low", "medium", "high", "very_high"], duplicates='drop')
            train_classification_models(X_processed, y_bucket, target_name)
            extra["classification_models"] = True
        except Exception as e:
            logger.warning(f"Classification training failed: {e}")
            extra["classification_models"] = False

    if target_name == "target_average_rating":
        huber = Pipeline([
            ("scaler", RobustScaler()),
            ("huber", HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=200)),
        ])
        huber.fit(X_cleaned.fillna(0), y_transformed)
        dump(huber, MODELS_DIR / f"huber_{target_name}.joblib")
        logger.info(f"Saved Huber model -> {MODELS_DIR / f'huber_{target_name}.joblib'}")
        extra.update({
            "rating_clip": [1.0, 5.0],
            "huber_params": huber.named_steps['huber'].get_params()
        })

    meta = {
        "target": target_name,
        "feature_columns": list(X.columns),
        "n_train_rows": int(X.shape[0]),
        "rf_params": rf.get_params(),
        **extra,
    }
    meta_path = MODELS_DIR / f"meta_{target_name}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
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

    print(f"\nTraining complete. Models stored in: {MODELS_DIR}")


if __name__ == "__main__":
    main()