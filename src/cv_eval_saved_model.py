import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("NULL")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved CV brain on a processed CSV.")
    parser.add_argument("--model", type=Path, required=True, help="Path to saved brain .joblib")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed CSV (must contain wf_* and gesture)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/models/cv_eval"))
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data not found: {args.data}")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    exclude_null = bool(bundle.get("exclude_null", False))

    df = pd.read_csv(args.data, low_memory=False)
    if "gesture" not in df.columns:
        raise ValueError("Evaluation CSV must contain 'gesture' column")

    y = normalize_label(df["gesture"])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if exclude_null:
        keep = y != "NULL"
        X = X[keep]
        y = y[keep]

    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="macro"))
    report = classification_report(y, y_pred, digits=4, zero_division=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_path": str(args.model),
        "data_path": str(args.data),
        "rows": int(len(X)),
        "accuracy": acc,
        "macro_f1": f1,
        "exclude_null": exclude_null,
    }
    with open(args.out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_dir / "eval_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    classes = sorted(pd.unique(y))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        display_labels=classes,
        values_format="d",
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    disp.ax_.set_title("CV Eval Confusion Matrix")
    disp.ax_.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(args.out_dir / "eval_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print("Rows evaluated:", len(X))
    print("Accuracy:", f"{acc:.4f}")
    print("Macro-F1:", f"{f1:.4f}")
    print("Saved:", args.out_dir / "eval_summary.json")
    print("Saved:", args.out_dir / "eval_report.txt")
    print("Saved:", args.out_dir / "eval_confusion_matrix.png")


if __name__ == "__main__":
    main()
