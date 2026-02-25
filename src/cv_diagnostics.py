import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_DATASET = Path("data/processed/dataset_geonorm_stdscaled.csv")
DEFAULT_OUT_DIR = Path("data/models/cv_diagnostics")


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("NULL")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def get_cv_cols(columns):
    r_cols = [c for c in columns if c.startswith("R_") and len(c) >= 4 and c[2] in ("x", "y", "z")]
    l_cols = [c for c in columns if c.startswith("L_") and len(c) >= 4 and c[2] in ("x", "y", "z")]
    if r_cols:
        return sorted(r_cols), "R"
    if l_cols:
        return sorted(l_cols), "L"
    raise ValueError("No CV landmark columns found.")


def top_confusions(cm: np.ndarray, classes: list[str], k: int = 10):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append((int(cm[i, j]), classes[i], classes[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:k]


def main():
    parser = argparse.ArgumentParser(description="CV diagnostics for gesture classification.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--exclude-null", action="store_true")
    parser.add_argument("--model", choices=["svm", "rf", "knn", "logreg"], default="svm")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    y = normalize_label(df["gesture"])

    cols, hand_mode = get_cv_cols(df.columns)
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if args.exclude_null:
        mask = y != "NULL"
        X = X[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = {
        "svm": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=args.random_state))]),
        "rf": RandomForestClassifier(n_estimators=300, random_state=args.random_state, n_jobs=-1),
        "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=args.random_state))]),
    }

    model = models[args.model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    classes = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    conf_pairs = top_confusions(cm, classes, k=12)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save text summary
    summary_path = args.out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Feature hand mode: {hand_mode}\n")
        f.write(f"Feature count: {len(cols)}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Exclude NULL: {args.exclude_null}\n")
        f.write(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
        f.write("Top confusion pairs (true -> predicted):\n")
        for n, t, p in conf_pairs:
            f.write(f"  {t} -> {p}: {n}\n")
        f.write("\nClassification report:\n")
        f.write(report)

    # Save confusion matrix figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_title(f"Confusion Matrix ({args.model.upper()})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {args.out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
