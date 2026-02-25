# mldm2

## Overview
This repository contains three tracks:
- **CV track** (active): landmark preprocessing, model training, evaluation.
- **Sensor track** (placeholder): glove/IMU pipeline.
- **Fusion track** (placeholder): early and late fusion integration.

Main folders:
- `src/` scripts
- `data/raw/` raw CSV and landmark captures
- `data/processed/` processed datasets
- `data/models/` trained models and reports
- `data/test/` unseen test data

---

## Environment
```bash
cd /home/lystiger/Documents/ML2/mldm2
source venv/bin/activate
pip install -r requirements.txt
```

---

## Script Catalog (Current)

### CV data and preprocessing
- `src/extract_cv_raw_subset.py`
  - Extracts target gestures into one raw CSV.
- `src/build_test_csv_from_folder.py`
  - Merges many raw test CSV files in `data/test/cv_raw/` into one file.
- `src/cv_preprocess_pipeline.py`
  - Builds:
    - frame-level processed CSV
    - window-level processed CSV (sliding windows)

### CV training, diagnostics, tuning
- `src/cv_diagnostics.py`
  - Single-model diagnostic (confusion + report).
- `src/cv_diagnostics_compare.py`
  - 4-model compare (RF/SVM/KNN/LogReg) + plots.
- `src/cv_temporal_diagnostics.py`
  - Temporal diagnostics with window features.
- `src/cv_window_svm_tune.py`
  - SVM hyperparameter sweep on window features + tuned plots.
- `src/cv_train_window_brain.py`
  - Trains and saves deployable CV brain (`.joblib`).
- `src/cv_eval_saved_model.py`
  - Evaluates saved brain on a processed test CSV.

### Sensor / Fusion scripts
- `src/collect_data.py`
- `src/late_fusion_runner.py`

---

## CV Pipeline (Recommended)

## 1) Build train raw CSV (selected classes)
```bash
venv/bin/python src/extract_cv_raw_subset.py \
  --root data/raw/landmark \
  --out data/raw/CV_raw_extracted.csv
```

## 2) Preprocess train CSV (frame + window)
```bash
venv/bin/python src/cv_preprocess_pipeline.py \
  --input data/raw/CV_raw_extracted.csv \
  --frame-out data/processed/cv_frame_processed.csv \
  --window-out data/processed/cv_window_processed.csv \
  --window 8 --step 4
```

## 3) Train CV brains (window-based)
```bash
# RF no-null (current strongest on unseen test)
venv/bin/python src/cv_train_window_brain.py \
  --data data/processed/cv_window_processed.csv \
  --model rf --exclude-null \
  --out data/models/cv_brains/cv_brain_window_rf_no_null.joblib

# RF with-null
venv/bin/python src/cv_train_window_brain.py \
  --data data/processed/cv_window_processed.csv \
  --model rf \
  --out data/models/cv_brains/cv_brain_window_rf_with_null.joblib
```

## 4) Build unseen test CSV from folder
```bash
venv/bin/python src/build_test_csv_from_folder.py \
  --root data/test/cv_raw \
  --out data/test/cv_raw/CV_test_raw_merged.csv
```

## 5) Preprocess unseen test CSV
```bash
venv/bin/python src/cv_preprocess_pipeline.py \
  --input data/test/cv_raw/CV_test_raw_merged.csv \
  --frame-out data/test/cv_processed/cv_test_frame.csv \
  --window-out data/test/cv_processed/cv_test_window.csv \
  --window 8 --step 4
```

## 6) Evaluate saved brains on unseen test
```bash
# RF no-null
venv/bin/python src/cv_eval_saved_model.py \
  --model data/models/cv_brains/cv_brain_window_rf_no_null.joblib \
  --data data/test/cv_processed/cv_test_window.csv \
  --out-dir data/models/cv_eval_test_rf_no_null

# RF with-null
venv/bin/python src/cv_eval_saved_model.py \
  --model data/models/cv_brains/cv_brain_window_rf_with_null.joblib \
  --data data/test/cv_processed/cv_test_window.csv \
  --out-dir data/models/cv_eval_test_rf_with_null
```

---

## Current CV Progress Snapshot
Data used:
- Train source: `data/raw/CV_raw_extracted.csv`
- Test source: `data/test/cv_raw/CV_test_raw_merged.csv`
- Feature mode: sliding-window (`window=8`, `step=4`)

Unseen test performance (latest):
- **RF no-null**: Accuracy `0.6238`, Macro-F1 `0.6134`  (best current baseline)
- RF with-null: Accuracy `0.5132`, Macro-F1 `0.5052`
- SVM no-null: Accuracy `0.4803`, Macro-F1 `0.4156`
- SVM with-null: Accuracy `0.3841`, Macro-F1 `0.3363`

Important:
- Internal holdout scores were higher; unseen test is the real benchmark.
- Current CV generalization is moderate; more data/cleanup/split rigor needed.

---

## Existing Plot Outputs
Window 4-model compare:
- `data/models/cv_window_processed_compare/accuracy_comparison_4_models.png`
- `data/models/cv_window_processed_compare/confusion_matrices_4_models.png`
- `data/models/cv_window_processed_compare/roc_curves_4_models.png`

Tuned SVM plots:
- `data/models/cv_window_svm_tune_no_null/svm_confusion_matrix.png`
- `data/models/cv_window_svm_tune_no_null/svm_roc_curve.png`
- `data/models/cv_window_svm_tune_with_null/svm_confusion_matrix.png`
- `data/models/cv_window_svm_tune_with_null/svm_roc_curve.png`

Model eval confusion plots:
- `data/models/cv_eval_test_rf_no_null/eval_confusion_matrix.png`
- `data/models/cv_eval_test_rf_with_null/eval_confusion_matrix.png`
- `data/models/cv_eval_test_svm_no_null/eval_confusion_matrix.png`
- `data/models/cv_eval_test_svm_with_null/eval_confusion_matrix.png`

---

## Placeholders (To Fill Later)

### Sensor Side TODO
- [ ] Finalize sensor dataset split policy (file/session level)
- [ ] Build sensor-only preprocessing pipeline
- [ ] Train and save sensor brains (`sensor_brain_*.joblib`)
- [ ] Add unseen sensor test reports under `data/models/sensor_eval_*`

### Fusion Side TODO
- [ ] Early fusion benchmark (timestamp merge quality checks)
- [ ] Late fusion benchmark on synchronized data (`pairing=timestamp`)
- [ ] Unified eval script for CV+Sensor on unseen sessions
- [ ] Final deployment config (`fusion_config.json`) with fixed weights and thresholds

---

## Notes
- Pandas `PerformanceWarning` about fragmented DataFrame is non-fatal.
- For reliable conclusions, always prioritize **unseen session** results over random split results.
