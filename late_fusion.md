# Unified Late Fusion Spec (CV + Glove)

This document merges the previous `late_fusion.md` and `late_fusion_update.md` into one canonical implementation spec.

## 1. Definition
This project uses **true late fusion (decision-level fusion)**.

- Train a **CV branch** and a **Glove branch** independently.
- Each branch outputs class probabilities.
- Final prediction is made by fusing branch probabilities.

Feature concatenation is **not** late fusion; it is early/feature-level fusion.

---

## 2. Data Sources (Current Repo)

### Branch A: Computer Vision
- File: `data/raw/all_gestures.csv`
- Label column: `gesture`
- Features: primarily right-hand landmarks `R_x0 ... R_z20` (optionally include hand-state flags)

### Branch B: Smart Glove
- File: `data/raw/glove_dataset.csv`
- Label column: `label`
- Features: `ax, ay, az, gx, gy, gz, f1, f2, f3, f4, f5`

### Label Normalization (Required)
Use one shared label format in both branches:
- trim spaces
- uppercase
- replace spaces with underscore

Example: `THANK YOU` -> `THANK_YOU`.

---

## 3. Temporal Synchronization (Event-Sync)
To align asynchronous camera and serial streams, use a physical sync event ("sync tap").

### 3.1 Spike Detection Defaults
- Glove spike: acceleration magnitude `x(t) > mu + 6*sigma` for `>= 2` samples.
- CV spike: wrist velocity `vf > v_mean + 5*sigma_v` for `>= 2` frames.

These are starting defaults; keep them configurable.

### 3.2 Offset Calibration
Compute initial offset:
\[
\Delta = t_{sensor\_sync} - t_{cv\_sync}
\]
Use `Delta` to align branch predictions.

### 3.3 Drift Management
Clock drift is expected. Add one:
- periodic re-sync (every `N` seconds or `K` gestures), or
- rolling offset update from recent matched events.

Log both `delta_initial` and `delta_runtime`.

---

## 4. Model Branches

### 4.1 CV Branch
1. Select CV features (`R_` default if left-hand is unreliable).
2. Preprocess: `StandardScaler` (+ optional `PCA(n_components=12)`).
3. Train candidate models (SVM/RF/KNN/LogReg).
4. Keep best CV model by validation metric.

Output:
- `P_cv(c)` for each class `c`.

### 4.2 Glove Branch
1. Use glove features (`ax..gz`, `f1..f5`).
2. Preprocess: `StandardScaler`.
3. Train candidate models (SVM/RF/KNN/LogReg).
4. Keep best glove model by validation metric.

Output:
- `P_glove(c)` for each class `c`.

---

## 5. Late Fusion Rules

### 5.1 Weighted Probability Fusion (Recommended)
\[
P_{fused}(c) = w_{cv} P_{cv}(c) + w_{glove} P_{glove}(c), \quad w_{cv}+w_{glove}=1
\]

Starting weights:
- `w_cv = 0.6`
- `w_glove = 0.4`

Tune on validation data.

Final class:
\[
\hat{y} = \arg\max_c P_{fused}(c)
\]

### 5.2 Product Rule (Optional)
\[
P_{fused}(c) \propto P_{cv}(c)^{w_{cv}} \cdot P_{glove}(c)^{w_{glove}}
\]
Use if branch probabilities are well-calibrated.

### 5.3 Probability Calibration (Recommended)
Calibrate branch probabilities before fusion:
- CV: Platt scaling or isotonic
- Glove: isotonic (or equivalent)

---

## 6. Runtime Decision Policy

### 6.1 Time Alignment at Inference
For each decision time `T`:
- use `P_cv` at `T`
- use `P_glove` at `T + Delta`

Initial acceptance target: `|Delta| <= 150 ms` (relax only if necessary).

### 6.2 Confidence Threshold
If `max(P_fused) > 0.75`, emit class. Otherwise use fallback.

### 6.3 Fallback for Missing/Low-Confidence Branches
- if CV strong and glove missing: use CV-only
- if glove strong and CV missing: use glove-only
- if both weak: output `unknown`/`null`

Suggested defaults:
- `tau_cv = 0.65`
- `tau_glove = 0.65`

---

## 7. Real-Time Execution Flow
1. Wait for sync event in both streams.
2. Compute `Delta` and validate.
3. Predict loop:
   - infer `P_cv`
   - infer aligned `P_glove`
   - calibrate (if available)
   - fuse probabilities
   - apply confidence + fallback policy
4. Log outputs to `data/logs/`.

---

## 8. Artifacts
Save to `data/models/`:
- `cv_model.joblib` + `cv_scaler.joblib` (+ optional `cv_pca.joblib`)
- `glove_model.joblib` + `glove_scaler.joblib`
- shared `label_encoder.joblib`
- fusion config (`weights`, thresholds, sync tolerance)

---

## 9. Implementation Checklist
- [ ] Normalize labels identically for CV and glove data.
- [ ] Train CV and glove branches independently.
- [ ] Ensure same class order in both probability outputs.
- [ ] Implement event-based sync offset `Delta`.
- [ ] Add drift handling (`delta_runtime`).
- [ ] Implement weighted probability fusion.
- [ ] Add calibration + fallback policy.
- [ ] Export branch models/scalers/encoder/config.
- [ ] Compare CV-only vs glove-only vs fused metrics (Accuracy, Macro-F1, CM).
