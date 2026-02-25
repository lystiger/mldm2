# CV Improvement Plan (For Late Fusion Project)

## Goal
Increase CV branch quality from current ~0.57 accuracy / ~0.62 macro-F1 to a robust level for fusion.

## Current Situation
- Dataset: `data/raw/CV_raw_extracted.csv`
- Main issue: many gesture samples collapse into `NULL`
- Symptom: high precision for gesture classes, but low recall (model predicts `NULL` too often)

---

## Priority Roadmap

## 1) Fix Evaluation Leakage First
Use **group split by take/source file** (not random row split).

Why:
- Random split can overestimate performance.
- Group split gives realistic generalization.

Success criteria:
- Stable metrics across seeds (low variance).

## 2) Handle `NULL` as a Separate Stage
Move from 1-stage multiclass to 2-stage classification:
- Stage A: `NULL` vs `GESTURE`
- Stage B: classify only gesture classes (`CALL, EAT, HELLO, IS, ME, THANK_YOU, YES`)

Why:
- `NULL` currently absorbs many gesture frames.

Success criteria:
- Gesture recall improves significantly while keeping false positives reasonable.

## 3) Use Temporal Features by Default
Use sliding windows and motion features:
- mean/std over window
- first-last delta
- velocity mean/std
- tune `window` in `{8, 12, 16}` and `step` in `{2, 4}`

Why:
- Gesture identity is temporal; frame-only landmarks are weak.

Success criteria:
- Temporal macro-F1 > frame-level macro-F1 under same split.

## 4) Strengthen Geometry Features
Add relative features (less camera-dependent):
- landmark-to-wrist vectors
- selected pairwise distances
- joint angle proxies
- scale normalization by hand size

Why:
- Raw xyz is sensitive to position and distance.

Success criteria:
- Better cross-session performance and fewer NULL collapses.

## 5) Improve Data Hygiene
- Remove transition frames at clip boundaries.
- Keep one consistent active hand for this setup.
- Audit confusing class pairs from confusion matrix.

Why:
- Label noise hurts recall and temporal consistency.

Success criteria:
- Fewer dominant confusion pairs over repeated runs.

---

## Experiment Order (Do This Sequence)
1. Baseline with group split (frame-level).
2. Add NULL 2-stage pipeline.
3. Add temporal windows + sweep (`window`, `step`).
4. Add geometry/relative features.
5. Retune model hyperparameters (RF/SVM/KNN/LogReg).
6. Lock best CV branch and re-run late fusion.

---

## Model Guidance
- Start with **Random Forest** for CV baseline (currently best in robust setting).
- Keep SVM as second candidate after feature improvements.
- Use macro-F1 as primary metric (not only accuracy).

---

## What "Good Enough" Looks Like Before Fusion
Minimum target before trusting fusion gains:
- CV macro-F1 >= 0.75 on robust split
- Stable confusion matrix across 3 random seeds
- Significant reduction in `gesture -> NULL` errors

---

## Deliverables
- Updated diagnostics plots (accuracy, confusion matrices, ROC)
- Best CV model + config
- Short report: what changed and metric impact per step
