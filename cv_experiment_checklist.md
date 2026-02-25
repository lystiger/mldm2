# CV Experiment Checklist

## Scope
Dataset: `data/raw/CV_raw_extracted.csv`

## Run Plan
- [ ] A1. Frame-level robust baseline (single-hand, with NULL, 4-model compare)
- [ ] A2. Frame-level robust without NULL (single-hand, 4-model compare)
- [ ] B1. Temporal baseline (SVM, window=8, step=4, with NULL)
- [ ] B2. Temporal baseline without NULL (SVM, window=8, step=4)
- [ ] B3. Temporal RF sweep (window=8/12/16, step=4)

## Commands
A1
```bash
venv/bin/python src/cv_diagnostics_compare.py \
  --data data/raw/CV_raw_extracted.csv \
  --feature-mode single \
  --out-dir data/models/cv_diag_compare_single
```

A2
```bash
venv/bin/python src/cv_diagnostics_compare.py \
  --data data/raw/CV_raw_extracted.csv \
  --feature-mode single \
  --exclude-null \
  --out-dir data/models/cv_diag_compare_single_no_null
```

B1
```bash
venv/bin/python src/cv_temporal_diagnostics.py \
  --data data/raw/CV_raw_extracted.csv \
  --model svm --window 8 --step 4 \
  --out-dir data/models/cv_temporal_diag_svm
```

B2
```bash
venv/bin/python src/cv_temporal_diagnostics.py \
  --data data/raw/CV_raw_extracted.csv \
  --model svm --window 8 --step 4 --exclude-null \
  --out-dir data/models/cv_temporal_diag_svm_no_null
```

B3 (repeat with different windows)
```bash
venv/bin/python src/cv_temporal_diagnostics.py --data data/raw/CV_raw_extracted.csv --model rf --window 8  --step 4 --out-dir data/models/cv_temporal_diag_rf_w8
venv/bin/python src/cv_temporal_diagnostics.py --data data/raw/CV_raw_extracted.csv --model rf --window 12 --step 4 --out-dir data/models/cv_temporal_diag_rf_w12
venv/bin/python src/cv_temporal_diagnostics.py --data data/raw/CV_raw_extracted.csv --model rf --window 16 --step 4 --out-dir data/models/cv_temporal_diag_rf_w16
```

## Decision Rule
- Pick the setting with highest **Macro-F1** while avoiding collapse into `NULL`.
- Prefer temporal setting if it improves Macro-F1 with stable confusion matrix.
