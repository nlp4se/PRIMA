from pathlib import Path

# Root project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# DATAR dataset paths
DATAR_DIR = INPUT_DIR / "DATAR"
RELEASE_JSON_DIR = DATAR_DIR / "release_related" / "all_jsons"
REVIEW_DIR = DATAR_DIR / "review_related"  # If reviews are separate

# Processed data paths
FILTERED_RELEASES_DIR = PROCESSED_DIR / "filtered_releases"
FEATURES_DIR = PROCESSED_DIR / "features"
DATASETS_DIR = PROCESSED_DIR / "datasets"

# Output paths
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Release filtering
MIN_RELEASES_PER_APP = 5  # Minimum releases per app to include
MIN_RELEASES_WITH_REVIEWS = 3  # Minimum releases with reviews per app

# Temporal validation
TEMPORAL_SPLIT_RATIO = 0.7


# todo
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}


# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = OUTPUT_DIR / 'prima_xgb.log'

