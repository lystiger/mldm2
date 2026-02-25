# Multi-Modal Late Fusion Strategy: CV + Glove Sensors

## 1. Data Architecture
This project uses **Late Fusion (Feature-Level)** to analyze hand gestures by combining spatial coordinates (Vision) with mechanical movement (Sensors).

### Data Stream A: Computer Vision (The "Visual Brain")
* **Source:** MediaPipe Hand Landmarks (`yes.csv`).
* **Input:** 63 features (Right Hand `R_x0, R_y0, R_z0` through `R_z20`).
* **Note:** Left hand columns (`L_`) are ignored as they contain null data in our dataset.
* **Pre-processing:** Scaling + PCA (Principal Component Analysis) to reduce 63 dimensions to a compact ~10-feature vector.

### Data Stream B: Smart Glove (The "Physical Brain")
* **Source:** 6-axis IMU + 5 Flex Sensors (`glove_datasett.csv`).
* **Input:** 11 features (`ax, ay, az, gx, gy, gz, f1, f2, f3, f4, f5`).
* **Pre-processing:** StandardScaler to normalize raw sensor units.

---

## 2. Fusion Workflow

### Step 1: Feature Extraction (Local Analysis)
We run the data through their respective "brains" simultaneously:
- **CV Feature Vector ($V_{cv}$):** Represents the *shape* and *position* of the hand.
- **Sensor Feature Vector ($V_{s}$):** Represents the *tension* and *motion* of the hand.

### Step 2: Concatenation (The Fusion)
The two vectors are "glued" together into a single global feature vector:
$$V_{fused} = [V_{cv} + V_{s}]$$

### Step 3: Global Analysis (The Executive)
The fused vector is fed into the following four classifiers to compare performance:
1. **KNN:** Measures similarity to known gesture clusters.
2. **SVM:** Finds the optimal hyperplane to separate complex gesture data.
3. **Logistic Regression:** Provides a baseline for linear separability.
4. **Random Forest:** Handles non-linear interactions between vision and sensors.

---

## 3. Implementation Checklist
- [ ] **Synchronization:** Align `timestamp_ms` from CV and Sensors.
- [ ] **Hand Filtering:** Ensure only `R_` landmarks are used if only the right hand is active.
- [ ] **Normalization:** Apply `StandardScaler` to the fused vector before feeding into SVM/KNN.
- [ ] **Model Persistence:** Export `cv_model.pkl` and `sensor_model.pkl` for real-time inference.