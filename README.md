# PRIMA: Forecasting Mobile App Release Impact with Metadata

A forecasting framework built on the [DATAR dataset](https://zenodo.org/records/10579421) to predict release success using **pre-release metadata** and XGBoost models, with proper temporal validation.

---

### 1. Installation
```bash
cd prima
pip install -r requirements.txt
````

---

### 2. Data Setup

Place the DATAR dataset **unzipped** in:

```
data/input/DATAR/release_related/all_jsons/
```

---

### 3. Run Full Pipeline

Run step by step:

```bash
python data_preprocessing/01_filter_data.py      # Filter releases with reviews (takes a while)
python data_preprocessing/02_extract_features.py # Extract feature sets from raw release metadata
python data_preprocessing/03_build_datasets.py   # Create temporally split train/test datasets
python data_preprocessing/04_data_quality.py     # Evaluate basic dataset structure and integrity
python data_preprocessing/05_feature_target_correlation.py  # RQ1: Analyze correlations between features and outcomes
python modeling/06_predict_release_impact.py     # RQ2: Train and evaluate forecasting models --> still in dev
```

---

### 4. Research Questions

#### RQ1: Measuring Correlation and Impact

**To what extent is there a correlation between release metrics and our target metrics (review count and average rating)?**

We compute Pearson correlations and Mutual Information (MI) scores between all numeric release metadata features and the target variables:

* `target_review_count` (future user engagement)
* `target_average_rating` (future user satisfaction)

We found:

* `target_review_count` positively correlates with features like `num_broadcast_receivers`, `total_components`, and `issue/pr counts`. This means that structural complexity and development activity attract user attention.
* `target_average_rating` shows negative correlation with `apk_file_size`, `num_permissions`, and `release_name_length`, suggesting that larger or more complex apps are perceived as lower quality.
* MI analysis shows non-linear but informative relationships with features such as `apk_arsc_size`, `min_sdk_version`, and `body_length`.

We can say thus that certain metadata fields contain predictive signals and can be used to guide feature selection and segmentation for downstream forecasting.

---
