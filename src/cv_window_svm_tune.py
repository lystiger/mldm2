import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC


DEFAULT_DATA = Path("data/processed/cv_window_processed.csv")
DEFAULT_OUT = Path("data/models/cv_window_svm_tune")


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("NULL")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SVM tuning on window-processed CV features.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exclude-null", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = pd.read_csv(args.data, low_memory=False)
    if "gesture" not in df.columns:
        raise ValueError("Missing 'gesture' column.")

    y = normalize_label(df["gesture"])
    X = df[[c for c in df.columns if c.startswith("wf_")]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if args.exclude_null:
        keep = y != "NULL"
        X = X[keep]
        y = y[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    grid = {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1, 3, 10, 30, 100],
        "gamma": ["scale", 0.1, 0.03, 0.01, 0.003, 0.001],
    }

    # gamma is only relevant for rbf; keep code simple and skip non-useful combos for linear
    candidates = []
    for kernel, C, gamma in itertools.product(grid["kernel"], grid["C"], grid["gamma"]):
        if kernel == "linear" and gamma != "scale":
            continue
        candidates.append({"kernel": kernel, "C": C, "gamma": gamma})

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.random_state)

    results = []
    best = None
    best_score = -1.0

    for p in candidates:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=p["kernel"],
                    C=p["C"],
                    gamma=p["gamma"],
                    probability=False,
                    random_state=args.random_state,
                ),
            ),
        ])

        fold_scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_va = y_train.iloc[va_idx]

            pipe.fit(X_tr, y_tr)
            yp = pipe.predict(X_va)
            fold_scores.append(f1_score(y_va, yp, average="macro"))

        cv_f1 = float(np.mean(fold_scores))
        results.append({"params": p, "cv_macro_f1": cv_f1})

        if cv_f1 > best_score:
            best_score = cv_f1
            best = p

    # Fit best on train and evaluate on held-out test.
    best_pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel=best["kernel"],
                C=best["C"],
                gamma=best["gamma"],
                probability=True,
                random_state=args.random_state,
            ),
        ),
    ])
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)

    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_dir / "svm_grid_results.json", "w", encoding="utf-8") as f:
        json.dump(sorted(results, key=lambda d: d["cv_macro_f1"], reverse=True), f, indent=2)

    summary = {
        "data": str(args.data),
        "exclude_null": args.exclude_null,
        "best_params": best,
        "best_cv_macro_f1": best_score,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    with open(args.out_dir / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots for the tuned SVM.
    classes = np.array(sorted(pd.unique(y_test)))
    y_test_bin = label_binarize(y_test, classes=classes)
    # Align proba to sorted class order.
    aligned = np.zeros((y_proba.shape[0], len(classes)))
    cidx = {c: i for i, c in enumerate(classes)}
    for i, c in enumerate(best_pipe.named_steps["svm"].classes_):
        aligned[:, cidx[c]] = y_proba[:, i]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=classes,
        values_format="d",
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    disp.ax_.set_title("Tuned SVM Confusion Matrix")
    disp.ax_.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(args.out_dir / "svm_confusion_matrix.png", dpi=200)
    plt.close(fig)

    # ROC micro-average
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), aligned.ravel())
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"Tuned SVM (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Tuned SVM ROC (Micro-average)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "svm_roc_curve.png", dpi=200)
    plt.close(fig)

    print("Best params:", best)
    print("CV macro-F1:", round(best_score, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("Test macro-F1:", round(test_f1, 4))
    print("Saved:", args.out_dir / "best_summary.json")
    print("Saved:", args.out_dir / "svm_confusion_matrix.png")
    print("Saved:", args.out_dir / "svm_roc_curve.png")


if __name__ == "__main__":
    main()
