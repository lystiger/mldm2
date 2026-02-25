from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    log_loss,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

# ==========================================
# 0. CONFIG
# ==========================================
DATA_PATH = Path("all_gestures.csv")
TARGET_COL = "gesture"
OUTPUT_DIR = Path("plots")
RANDOM_STATE = 42


def show_or_save_plot(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_path.resolve()}")
    plt.close(fig)


def align_proba_to_classes(y_proba: np.ndarray, model_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((y_proba.shape[0], len(all_classes)))
    class_to_idx = {name: idx for idx, name in enumerate(all_classes)}
    for i, name in enumerate(model_classes):
        aligned[:, class_to_idx[name]] = y_proba[:, i]
    return aligned


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cannot find dataset: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH, keep_default_na=False)
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}.")

feature_cols = [c for c in df.columns if c.startswith("L_") or c.startswith("R_")]
if not feature_cols:
    raise ValueError("No feature columns found with L_/R_ prefixes.")

# Include unlabeled rows as explicit 'null' class.
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
df[TARGET_COL] = df[TARGET_COL].replace({"": "null", "nan": "null", "NaN": "null", "None": "null"})

# Keep only rows with valid numeric features.
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
before_drop = len(df)
df = df.dropna(subset=feature_cols).copy()
dropped_rows = before_drop - len(df)

print(f"Loaded data: {DATA_PATH.resolve()}")
print(f"Rows: {len(df)} | Features: {len(feature_cols)} | Classes: {df[TARGET_COL].nunique()}")
if dropped_rows:
    print(f"Dropped rows with invalid numeric features: {dropped_rows}")

X = df[feature_cols].to_numpy(dtype=np.float32)
y_text = df[TARGET_COL].to_numpy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
class_names = label_encoder.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Scale for SVM/KNN/LogReg.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced_subsample"),
        "scaled": False,
    },
    "SVM (RBF Kernel)": {
        "model": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "scaled": True,
    },
    "KNN (K=5)": {
        "model": KNeighborsClassifier(n_neighbors=5),
        "scaled": True,
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=4000, random_state=RANDOM_STATE, class_weight="balanced"),
        "scaled": True,
    },
}

print("\nStarting 4-model comparison...\n")

results = {}
macro_f1_scores = {}
losses = {}
loss_percentages = {}
roc_data = {}
conf_mats = {}
reports = {}

y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

for name, spec in models.items():
    model = spec["model"]
    if spec["scaled"]:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    y_proba_aligned = align_proba_to_classes(y_proba, model.classes_, np.arange(len(class_names)))

    acc = accuracy_score(y_test, y_pred) * 100
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    loss = log_loss(y_test, y_proba_aligned, labels=np.arange(len(class_names)))
    loss_pct = loss * 100.0

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba_aligned.ravel())
    roc_auc = auc(fpr, tpr)

    results[name] = acc
    macro_f1_scores[name] = macro_f1
    losses[name] = loss
    loss_percentages[name] = loss_pct
    roc_data[name] = (fpr, tpr, roc_auc)
    conf_mats[name] = (y_test, y_pred)
    reports[name] = classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0)

    print(
        f"{name:20s} | Accuracy: {acc:.2f}% | Macro-F1: {macro_f1:.4f} | "
        f"LogLoss: {loss:.4f} ({loss_pct:.2f}%) | AUC: {roc_auc:.4f}"
    )

print("\nAverage Metrics")
print(f"Average Accuracy: {np.mean(list(results.values())):.2f}%")
print(f"Average Macro-F1: {np.mean(list(macro_f1_scores.values())):.4f}")
print(f"Average Log Loss: {np.mean(list(losses.values())):.4f}")
print(f"Average Loss %: {np.mean(list(loss_percentages.values())):.2f}%")

best_model_name = max(macro_f1_scores, key=macro_f1_scores.get)
print(f"\nBest model by Macro-F1: {best_model_name}")
print(reports[best_model_name])

# Accuracy bar chart
fig = plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), color=["#4CAF50", "#2196F3", "#FFC107", "#E91E63"])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom")
plt.ylim(0, 110)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison: RF vs SVM vs KNN(k=5) vs Logistic")
plt.xticks(rotation=12, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
show_or_save_plot(fig, "accuracy_comparison_4_models.png")

# Confusion matrices (normalized)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.ravel()
for idx, (name, (y_true, y_pred)) in enumerate(conf_mats.items()):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        values_format="d",
        cmap="Blues",
        ax=axes[idx],
        colorbar=False,
    )
    disp.ax_.set_title(name)
    disp.ax_.tick_params(axis="x", rotation=45)
plt.suptitle("Normalized Confusion Matrices (4 Models)", fontsize=14, fontweight="bold")
plt.tight_layout()
show_or_save_plot(fig, "confusion_matrices_4_models.png")

# ROC micro-average comparison
fig = plt.figure(figsize=(10, 7))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (micro-average, 4 models)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
show_or_save_plot(fig, "roc_curves_4_models.png")
