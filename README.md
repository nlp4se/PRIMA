# PRIMA: Forecasting Mobile App Release Impact with Metadata

A forecasting framework built on the [DATAR dataset](https://zenodo.org/records/10579421) to predict release success using **pre-release metadata** and XGBoost models, with proper temporal validation.

---

### 1. Installation
```bash
cd prima
pip install -r requirements.txt
```

### 2. Data Setup
Place the DATAR dataset **unizpped** in `data/input/DATAR/release_related/all_jsons/`

### 3. Run Full Pipeline
run step by step:
```bash
python data_preprocessing/01_filter_data.py      # Filter releases with reviews --> takes a bit long!
python data_preprocessing/02_extract_features.py # Extract all features
python data_preprocessing/03_build_datasets.py   # Create datasets with temporal splits
python data_preprocessing/04_data_quality.py   # Evaluate datasets quality before model training 
```

---