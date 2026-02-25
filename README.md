# mldm2

## 1. What This Repo Does
This repo supports:
- CV landmark collection and processing
- Glove sensor collection
- Early fusion (feature/table merge)
- Late fusion (decision/probability fusion)
- CV diagnostics and model comparison

Core data paths:
- Raw: `data/raw/`
- Processed: `data/processed/`
- Model outputs: `data/models/`

---

## 2. Setup
From repo root:

```bash
cd /home/lystiger/Documents/ML2/mldm2
source venv/bin/activate
pip install -r requirements.txt
```

If `python` is not the venv interpreter, use `venv/bin/python` explicitly in commands below.

---

## 3. Script Index
- `src/collect_data.py`: collect glove/CV and fuse by timestamp
- `src/extract_cv_raw_subset.py`: extract selected gesture classes from `data/raw/landmark/`
- `src/cv_preprocess_pipeline.py`: CV preprocessing (relative normalize + window features)
- `src/cv_diagnostics.py`: single-model CV diagnostics
- `src/cv_diagnostics_compare.py`: 4-model CV compare + plots
- `src/cv_temporal_diagnostics.py`: temporal CV diagnostics with sliding windows
- `src/cv_window_svm_tune.py`: SVM hyperparameter tuning for window CV features
- `src/late_fusion_runner.py`: late fusion training/evaluation (label or timestamp pairing)
- `src/all_models_new.py`: legacy 4-model compare script (expects `all_gestures.csv` in current dir)

---

## 4. Data Collection Test Cases

## 4.1 Glove Only
```bash
python src/collect_data.py collect \
  --port /dev/ttyACM0 --baud 115200 --samples 100 \
  --out data/raw/glove_dataset.csv
```
What happens:
- Opens serial port
- Reads glove rows
- Writes `data/raw/glove_dataset.csv`

## 4.2 CV Only
```bash
python src/collect_data.py collect-cv \
  --out data/raw/hand_landmarks.csv --camera-id 0
```
What happens:
- Opens webcam
- `c` toggles recording, `1..9` sets gesture, `q` quits
- Writes `data/raw/hand_landmarks.csv`

## 4.3 Sync Collect (Best for true runtime fusion)
```bash
python src/collect_data.py collect-sync \
  --port /dev/ttyACM0 --baud 115200 --camera-id 0 \
  --glove-out data/raw/glove_dataset.csv \
  --camera-out data/raw/hand_landmarks.csv
```
What happens:
- Starts glove + CV together
- Same toggle keys in one flow
- Best option for true timestamp fusion later

---

## 5. Fusion Test Cases

## 5.1 Early Fusion (table merge by timestamp)
```bash
python src/collect_data.py fuse \
  --glove data/raw/glove_dataset.csv \
  --camera data/raw/hand_landmarks.csv \
  --out data/processed/fused_dataset.csv \
  --tol-ms 30
```
What happens:
- `merge_asof` nearest-time alignment
- Writes one fused dataset table

Failure mode:
- If clocks are unrelated, you get very few/zero aligned rows.

## 5.2 Late Fusion (decision-level), label pairing
```bash
python src/late_fusion_runner.py \
  --pairing label \
  --cv data/processed/dataset_geonorm_stdscaled.csv \
  --glove data/raw/glove_dataset.csv \
  --w-cv 0.6 \
  --out-dir data/models/late_fusion_label
```
What happens:
- Trains CV branch + glove branch separately
- Fuses probabilities
- Saves:
  - `cv_model.joblib`
  - `glove_model.joblib`
  - `fusion_config.json`
  - `metrics.json`

Use when glove hardware is unavailable for synced capture.

## 5.3 Late Fusion (true timestamp mode)
```bash
python src/late_fusion_runner.py \
  --pairing timestamp \
  --tolerance-ms 50 \
  --cv data/raw/all_gestures.csv \
  --glove data/raw/glove_dataset.csv \
  --out-dir data/models/late_fusion_timestamp
```
What happens:
- Uses timestamp alignment
- Requires synchronized captures

Failure mode:
- If timestamp bases differ, run fails with no aligned rows.

---

## 6. CV Improvement Pipeline (Recommended)
This is the current best path for CV quality.

## Step A: Extract only target classes
```bash
python src/extract_cv_raw_subset.py \
  --root data/raw/landmark \
  --out data/raw/CV_raw_extracted.csv
```
Expected:
- Combined CSV for `CALL,EAT,HELLO,IS,ME,NULL,THANK_YOU,YES`

## Step B: Preprocess CV data
```bash
python src/cv_preprocess_pipeline.py \
  --input data/raw/CV_raw_extracted.csv \
  --frame-out data/processed/cv_frame_processed.csv \
  --window-out data/processed/cv_window_processed.csv \
  --window 8 --step 4
```
Expected:
- `cv_frame_processed.csv` (relative normalized frame features)
- `cv_window_processed.csv` (temporal window features)

Note:
- `PerformanceWarning: DataFrame is highly fragmented` is non-fatal.

## Step C: 4-model compare on frame features
```bash
python src/cv_diagnostics_compare.py \
  --data data/processed/cv_frame_processed.csv \
  --feature-mode single \
  --out-dir data/models/cv_frame_processed_single
```
Outputs:
- accuracy bar, confusion matrix panel, ROC panel, summary

## Step D: 4-model compare on window features
(Already generated in this repo)
- `data/models/cv_window_processed_compare/accuracy_comparison_4_models.png`
- `data/models/cv_window_processed_compare/confusion_matrices_4_models.png`
- `data/models/cv_window_processed_compare/roc_curves_4_models.png`
- `data/models/cv_window_processed_compare/metrics.json`

## Step E: Tune SVM on window features
```bash
python src/cv_window_svm_tune.py \
  --data data/processed/cv_window_processed.csv \
  --exclude-null \
  --out-dir data/models/cv_window_svm_tune_no_null
```
Outputs:
- `best_summary.json`
- `svm_confusion_matrix.png`
- `svm_roc_curve.png`

---

## 7. Why Some Runs Look "Too Good"
If a result is unexpectedly near-perfect:
- check split strategy (random frame split may leak sequence similarity)
- check pairing mode (`label` pairing is prototype, not runtime sync)
- verify class collapse patterns (`NULL` domination)

Prefer metrics from:
- window features
- robust split settings
- explicit confusion matrix inspection

---

## 8. Optimal Workflow (Current)
If your goal is best offline classification now:
1. `extract_cv_raw_subset.py`
2. `cv_preprocess_pipeline.py` (window=8, step=4)
3. `cv_diagnostics_compare.py` on window-processed (or keep existing compare outputs)
4. `cv_window_svm_tune.py` (and compare against RF)
5. pick best CV model for late fusion

If your goal is production-like late fusion later:
1. collect synced data via `collect-sync`
2. run `late_fusion_runner.py --pairing timestamp`
3. validate with strict split policy

---

## 9. Quick Commands (Copy/Paste)

```bash
# Extract classes
venv/bin/python src/extract_cv_raw_subset.py --root data/raw/landmark --out data/raw/CV_raw_extracted.csv

# Preprocess frame + window
venv/bin/python src/cv_preprocess_pipeline.py --input data/raw/CV_raw_extracted.csv --frame-out data/processed/cv_frame_processed.csv --window-out data/processed/cv_window_processed.csv --window 8 --step 4

# Compare 4 models on frame processed
venv/bin/python src/cv_diagnostics_compare.py --data data/processed/cv_frame_processed.csv --feature-mode single --out-dir data/models/cv_frame_processed_single

# Tune SVM on window processed
venv/bin/python src/cv_window_svm_tune.py --data data/processed/cv_window_processed.csv --exclude-null --out-dir data/models/cv_window_svm_tune_no_null

# Label-based late fusion prototype
venv/bin/python src/late_fusion_runner.py --pairing label --cv data/processed/dataset_geonorm_stdscaled.csv --glove data/raw/glove_dataset.csv --w-cv 0.6 --out-dir data/models/late_fusion_label
```
