from pathlib import Path
import json
import logging
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from joblib import dump
from xgboost import XGBRegressor
import xgboost as xgb

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

def grid_iter(param_grid: dict):
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
        "lambda": cfg["reg_lambda"],
        "eta": cfg["learning_rate"],
        "seed": 42,
        "nthread": -1,
    }
    return p

def xgb_cv_select(X, y, objective, metric, base, grid, nfold=3, num_boost_round=2000, es_rounds=100):
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    dtrain = xgb.DMatrix(X, label=y)
    best = {"score": float("inf"), "params": None, "best_round": None}
    for params in grid_iter(grid):
        cfg = {**base, **params}
        cv_params = to_cv_params(cfg, objective)
        res = xgb.cv(
            params=cv_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            nfold=nfold,
            metrics=metric,
            early_stopping_rounds=es_rounds,
            verbose_eval=False,
            shuffle=True,
            seed=42,
        )
        best_round = len(res)
        score = float(res.iloc[-1][f"test-{metric}-mean"])
        if score < best["score"]:
            best.update({"score": score, "params": cfg, "best_round": best_round})
    final = XGBRegressor(
        n_estimators=best["best_round"],
        learning_rate=best["params"]["learning_rate"],
        max_depth=best["params"]["max_depth"],
        subsample=best["params"]["subsample"],
        colsample_bytree=best["params"]["colsample_bytree"],
        reg_lambda=best["params"]["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective=objective,
    )
    final.fit(X, y)
    return final, best

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

    base = dict(
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    grid = {
        "learning_rate": [0.03, 0.06, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [1.0, 5.0],
    }

    xgb_sq, best_sq = xgb_cv_select(
        X, y,
        objective="reg:squarederror",
        metric="rmse",
        base=base, grid=grid,
        nfold=3, num_boost_round=2000, es_rounds=100
    )
    dump(xgb_sq, MODELS_DIR / f"xgb_{target_name}.joblib")
    logger.info(f"Saved XGB model -> {MODELS_DIR / f'xgb_{target_name}.joblib'}")

    extra = {
        "xgb_best_params": best_sq["params"],
        "xgb_cv_best_round": int(best_sq["best_round"]),
        "xgb_cv_best_score": float(best_sq["score"]),
    }

    if target_name == "target_review_count":
        xgb_pois, best_pois = xgb_cv_select(
            X, y,
            objective="count:poisson",
            metric="poisson-nloglik",
            base=base, grid=grid,
            nfold=3, num_boost_round=2000, es_rounds=100
        )
        dump(xgb_pois, MODELS_DIR / f"xgb_poisson_{target_name}.joblib")
        logger.info(f"Saved XGB Poisson model -> {MODELS_DIR / f'xgb_poisson_{target_name}.joblib'}")

        y_log = np.log1p(y)
        xgb_log, best_log = xgb_cv_select(
            X, y_log,
            objective="reg:squarederror",
            metric="rmse",
            base=base, grid=grid,
            nfold=3, num_boost_round=2000, es_rounds=100
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

    if target_name == "target_average_rating":
        huber = Pipeline([
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(epsilon=1.35, alpha=0.0001)),
        ])
        huber.fit(X.fillna(0), y)
        dump(huber, MODELS_DIR / f"huber_{target_name}.joblib")
        logger.info(f"Saved Huber model -> {MODELS_DIR / f'huber_{target_name}.joblib'}")
        extra.update({"rating_clip": [1.0, 5.0]})

    meta = {
        "target": target_name,
        "feature_columns": list(X.columns),
        "n_train_rows": int(X.shape[0]),
        "rf_params": rf.get_params(),
        **extra,
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
