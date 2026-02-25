import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("NULL")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def build_model(name: str):
    if name == "svm_linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=3, probability=True, random_state=42)),
        ])
    if name == "svm_linear_with_null":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=30, probability=True, random_state=42)),
        ])
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    if name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ])
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42)),
        ])
    raise ValueError(f"Unsupported model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save CV brain from window-processed features.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/cv_window_processed.csv"))
    parser.add_argument("--model", choices=["svm_linear", "svm_linear_with_null", "rf", "knn", "logreg"], default="svm_linear")
    parser.add_argument("--exclude-null", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/models/cv_brains/cv_brain_window.joblib"))
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = pd.read_csv(args.data, low_memory=False)
    if "gesture" not in df.columns:
        raise ValueError("Dataset must contain 'gesture' column")

    y = normalize_label(df["gesture"])
    feature_cols = [c for c in df.columns if c.startswith("wf_")]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if args.exclude_null:
        keep = y != "NULL"
        X = X[keep]
        y = y[keep]

    # holdout estimate (for sanity)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(args.model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro"))

    # retrain on full dataset for deployment brain
    model.fit(X, y)

    bundle = {
        "model": model,
        "model_name": args.model,
        "feature_cols": feature_cols,
        "label_normalization": "upper_trim_space_to_underscore",
        "exclude_null": bool(args.exclude_null),
        "train_rows": int(len(X)),
        "classes": sorted(pd.unique(y).tolist()),
        "holdout_accuracy": acc,
        "holdout_macro_f1": f1,
        "source_data": str(args.data),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)

    summary = {
        "model": args.model,
        "exclude_null": bool(args.exclude_null),
        "holdout_accuracy": acc,
        "holdout_macro_f1": f1,
        "classes": bundle["classes"],
        "train_rows": bundle["train_rows"],
        "saved": str(args.out),
    }
    with open(args.out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved brain:", args.out)
    print("Holdout accuracy:", f"{acc:.4f}")
    print("Holdout macro-F1:", f"{f1:.4f}")


if __name__ == "__main__":
    main()
