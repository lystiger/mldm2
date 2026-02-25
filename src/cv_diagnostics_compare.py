import argparse
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
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC


DEFAULT_DATA = Path("data/raw/CV_raw_extracted.csv")
DEFAULT_OUT = Path("data/models/cv_diagnostics_compare")


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("NULL")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def pick_feature_cols(df: pd.DataFrame, mode: str) -> tuple[list[str], str]:
    l_cols = [c for c in df.columns if c.startswith("L_") and len(c) >= 4 and c[2] in ("x", "y", "z")]
    r_cols = [c for c in df.columns if c.startswith("R_") and len(c) >= 4 and c[2] in ("x", "y", "z")]

    if mode == "both":
        cols = sorted(l_cols + r_cols)
        hand = "L+R"
    else:
        # robust default: use one consistent hand branch
        if l_cols and not r_cols:
            cols = sorted(l_cols)
            hand = "L"
        elif r_cols and not l_cols:
            cols = sorted(r_cols)
            hand = "R"
        elif l_cols and r_cols:
            # choose hand with fewer missing values
            l_missing = df[l_cols].isna().sum().sum()
            r_missing = df[r_cols].isna().sum().sum()
            if l_missing <= r_missing:
                cols = sorted(l_cols)
                hand = "L"
            else:
                cols = sorted(r_cols)
                hand = "R"
        else:
            cols = []
            hand = "NONE"

    if not cols:
        raise ValueError("No valid landmark feature columns found.")
    return cols, hand


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CV models with robust settings and all-model style plots.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-mode", choices=["single", "both"], default="single")
    parser.add_argument("--exclude-null", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = pd.read_csv(args.data, low_memory=False)
    if "gesture" not in df.columns:
        raise ValueError("Dataset must contain 'gesture' column.")

    y = normalize_label(df["gesture"])
    cols, hand_mode = pick_feature_cols(df, args.feature_mode)

    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if args.exclude_null:
        keep = y != "NULL"
        X = X[keep]
        y = y[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=300, random_state=args.random_state, n_jobs=-1),
            "scaled": False,
        },
        "SVM (RBF)": {
            "model": SVC(kernel="rbf", probability=True, random_state=args.random_state),
            "scaled": True,
        },
        "KNN (K=5)": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "scaled": True,
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=3000, random_state=args.random_state),
            "scaled": True,
        },
    }

    class_names = np.array(sorted(pd.unique(y_test)))
    y_test_bin = label_binarize(y_test, classes=class_names)

    results = {}
    macro_f1 = {}
    reports = {}
    conf_data = {}
    roc_data = {}

    for name, spec in models.items():
        model = spec["model"]
        if spec["scaled"]:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        # align probability columns to class_names
        model_classes = model.classes_
        aligned = np.zeros((y_proba.shape[0], len(class_names)))
        idx = {c: i for i, c in enumerate(class_names)}
        for i, c in enumerate(model_classes):
            aligned[:, idx[c]] = y_proba[:, i]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), aligned.ravel())
        roc_auc = auc(fpr, tpr)

        results[name] = acc * 100
        macro_f1[name] = f1
        reports[name] = classification_report(y_test, y_pred, digits=4, zero_division=0)
        conf_data[name] = (y_test, y_pred)
        roc_data[name] = (fpr, tpr, roc_auc)

    ensure_out_dir(args.out_dir)

    # 1) Accuracy bar chart
    fig = plt.figure(figsize=(10, 6))
    names = list(results.keys())
    vals = [results[n] for n in names]
    bars = plt.bar(names, vals, color=["#4CAF50", "#2196F3", "#FFC107", "#E91E63"])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom")
    plt.ylim(0, 110)
    plt.title("CV Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(args.out_dir / "accuracy_comparison_4_models.png", dpi=200)
    plt.close(fig)

    # 2) Confusion matrices (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, name in enumerate(names):
        y_true, y_pred = conf_data[name]
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=class_names,
            values_format="d",
            cmap="Blues",
            ax=axes[i],
            colorbar=False,
        )
        disp.ax_.set_title(name)
        disp.ax_.tick_params(axis="x", rotation=45)
    plt.suptitle("Confusion Matrices (4 Models)", fontsize=14)
    plt.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrices_4_models.png", dpi=200)
    plt.close(fig)

    # 3) ROC curves
    fig = plt.figure(figsize=(10, 7))
    for name in names:
        fpr, tpr, roc_auc = roc_data[name]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Micro-average)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "roc_curves_4_models.png", dpi=200)
    plt.close(fig)

    # text summary
    with open(args.out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Feature mode: {args.feature_mode} (selected hand: {hand_mode})\n")
        f.write(f"Feature count: {len(cols)}\n")
        f.write(f"Exclude NULL: {args.exclude_null}\n")
        f.write(f"Train/Test: {len(X_train)}/{len(X_test)}\n\n")
        for name in names:
            f.write(f"{name}: Accuracy={results[name]:.2f}% | Macro-F1={macro_f1[name]:.4f}\n")
        f.write("\n")
        best = max(macro_f1, key=macro_f1.get)
        f.write(f"Best model by Macro-F1: {best}\n\n")
        f.write(reports[best])

    print(f"Saved: {args.out_dir / 'accuracy_comparison_4_models.png'}")
    print(f"Saved: {args.out_dir / 'confusion_matrices_4_models.png'}")
    print(f"Saved: {args.out_dir / 'roc_curves_4_models.png'}")
    print(f"Saved: {args.out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
