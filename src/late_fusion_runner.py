import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_CV_PATH = Path("data/processed/dataset_geonorm_stdscaled.csv")
DEFAULT_GLOVE_PATH = Path("data/raw/glove_dataset.csv")
DEFAULT_OUT_DIR = Path("data/models/late_fusion")


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def get_cv_feature_columns(columns):
    cv_cols = [c for c in columns if c.startswith("R_") and len(c) >= 4 and c[2] in ("x", "y", "z")]
    if not cv_cols:
        cv_cols = [c for c in columns if c.startswith("L_") and len(c) >= 4 and c[2] in ("x", "y", "z")]
    if not cv_cols:
        raise ValueError("No hand landmark columns found (R_x*/R_y*/R_z* or L_x*/L_y*/L_z*).")
    return sorted(cv_cols)


def preprocess_streams(cv_path: Path, glove_path: Path, pairing: str):
    if not cv_path.exists():
        raise FileNotFoundError(f"Missing CV dataset: {cv_path}")
    if not glove_path.exists():
        raise FileNotFoundError(f"Missing glove dataset: {glove_path}")

    cv = pd.read_csv(cv_path)
    glove = pd.read_csv(glove_path)

    required_cv = {"gesture"}
    if pairing == "timestamp":
        required_cv.add("timestamp_ms")
    required_glove = {"timestamp_ms", "label", "ax", "ay", "az", "gx", "gy", "gz", "f1", "f2", "f3", "f4", "f5"}

    if not required_cv.issubset(cv.columns):
        raise ValueError(f"CV dataset missing columns: {sorted(required_cv - set(cv.columns))}")
    if not required_glove.issubset(glove.columns):
        raise ValueError(f"Glove dataset missing columns: {sorted(required_glove - set(glove.columns))}")

    cv_cols = get_cv_feature_columns(cv.columns)
    glove_cols = ["ax", "ay", "az", "gx", "gy", "gz", "f1", "f2", "f3", "f4", "f5"]

    cv_base_cols = ["gesture"] + cv_cols
    if "timestamp_ms" in cv.columns:
        cv_base_cols = ["timestamp_ms"] + cv_base_cols
    cv_small = cv[cv_base_cols].copy()
    glove_small = glove[["timestamp_ms", "label"] + glove_cols].copy()

    cv_small["label_norm"] = normalize_label(cv_small["gesture"])
    glove_small["label_norm"] = normalize_label(glove_small["label"])

    if "timestamp_ms" in cv_small.columns:
        cv_small["timestamp_ms"] = pd.to_numeric(cv_small["timestamp_ms"], errors="coerce")
    glove_small["timestamp_ms"] = pd.to_numeric(glove_small["timestamp_ms"], errors="coerce")

    for col in cv_cols:
        cv_small[col] = pd.to_numeric(cv_small[col], errors="coerce")
    for col in glove_cols:
        glove_small[col] = pd.to_numeric(glove_small[col], errors="coerce")

    cv_drop_cols = cv_cols + ["label_norm"]
    if "timestamp_ms" in cv_small.columns:
        cv_drop_cols = ["timestamp_ms"] + cv_drop_cols
    cv_small = cv_small.dropna(subset=cv_drop_cols).copy()
    glove_small = glove_small.dropna(subset=["timestamp_ms"] + glove_cols + ["label_norm"]).copy()

    return cv_small, glove_small, cv_cols, glove_cols


def pair_by_timestamp(cv_small: pd.DataFrame, glove_small: pd.DataFrame, cv_cols, glove_cols, tolerance_ms: int):
    aligned = pd.merge_asof(
        cv_small.sort_values("timestamp_ms"),
        glove_small.sort_values("timestamp_ms"),
        on="timestamp_ms",
        direction="nearest",
        tolerance=tolerance_ms,
        suffixes=("_cv", "_glove"),
    )

    aligned = aligned.dropna(subset=["label_norm_cv", "label_norm_glove"] + cv_cols + glove_cols).copy()
    aligned = aligned[aligned["label_norm_cv"] == aligned["label_norm_glove"]].copy()

    if aligned.empty:
        raise ValueError("No aligned rows after timestamp merge and label matching. Increase tolerance or use --pairing label.")

    aligned["target"] = aligned["label_norm_cv"]
    return aligned


def pair_by_label(cv_small: pd.DataFrame, glove_small: pd.DataFrame, cv_cols, glove_cols, random_state: int):
    common_labels = sorted(set(cv_small["label_norm"].unique()) & set(glove_small["label_norm"].unique()))
    if not common_labels:
        raise ValueError("No common labels between CV and glove datasets.")

    rng = np.random.RandomState(random_state)
    chunks = []

    for label in common_labels:
        cv_g = cv_small[cv_small["label_norm"] == label].copy().reset_index(drop=True)
        gl_g = glove_small[glove_small["label_norm"] == label].copy().reset_index(drop=True)

        n = min(len(cv_g), len(gl_g))
        if n < 2:
            continue

        cv_idx = rng.permutation(len(cv_g))[:n]
        gl_idx = rng.permutation(len(gl_g))[:n]

        cv_s = cv_g.iloc[cv_idx].reset_index(drop=True)
        gl_s = gl_g.iloc[gl_idx].reset_index(drop=True)

        chunk = pd.DataFrame({
            "target": [label] * n,
            "timestamp_ms_cv": cv_s["timestamp_ms"] if "timestamp_ms" in cv_s.columns else np.nan,
            "timestamp_ms_glove": gl_s["timestamp_ms"],
        })

        for c in cv_cols:
            chunk[c] = cv_s[c].to_numpy()
        for c in glove_cols:
            chunk[c] = gl_s[c].to_numpy()

        chunks.append(chunk)

    if not chunks:
        raise ValueError("Could not create label-based pairs. Check class counts per label.")

    paired = pd.concat(chunks, axis=0, ignore_index=True)
    return paired


def fit_and_evaluate(aligned: pd.DataFrame, cv_cols: list[str], glove_cols: list[str], w_cv: float, random_state: int):
    y = aligned["target"].to_numpy()
    X_cv = aligned[cv_cols].to_numpy(dtype=np.float32)
    X_glove = aligned[glove_cols].to_numpy(dtype=np.float32)

    idx = np.arange(len(aligned))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    y_train = y[idx_train]
    y_test = y[idx_test]

    cv_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
    ])

    glove_model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    cv_model.fit(X_cv[idx_train], y_train)
    glove_model.fit(X_glove[idx_train], y_train)

    proba_cv = cv_model.predict_proba(X_cv[idx_test])
    proba_glove = glove_model.predict_proba(X_glove[idx_test])

    classes_cv = cv_model.classes_
    classes_glove = glove_model.classes_

    if list(classes_cv) != list(classes_glove):
        all_classes = sorted(set(classes_cv) | set(classes_glove))

        def align_proba(proba, model_classes, final_classes):
            out = np.zeros((proba.shape[0], len(final_classes)), dtype=np.float32)
            pos = {c: i for i, c in enumerate(final_classes)}
            for i, c in enumerate(model_classes):
                out[:, pos[c]] = proba[:, i]
            return out

        proba_cv = align_proba(proba_cv, classes_cv, all_classes)
        proba_glove = align_proba(proba_glove, classes_glove, all_classes)
        classes = np.array(all_classes)
    else:
        classes = classes_cv

    w_glove = 1.0 - w_cv
    proba_fused = w_cv * proba_cv + w_glove * proba_glove

    y_pred_cv = classes[np.argmax(proba_cv, axis=1)]
    y_pred_glove = classes[np.argmax(proba_glove, axis=1)]
    y_pred_fused = classes[np.argmax(proba_fused, axis=1)]

    metrics = {
        "cv_only": {
            "accuracy": float(accuracy_score(y_test, y_pred_cv)),
            "macro_f1": float(f1_score(y_test, y_pred_cv, average="macro")),
        },
        "glove_only": {
            "accuracy": float(accuracy_score(y_test, y_pred_glove)),
            "macro_f1": float(f1_score(y_test, y_pred_glove, average="macro")),
        },
        "late_fused": {
            "accuracy": float(accuracy_score(y_test, y_pred_fused)),
            "macro_f1": float(f1_score(y_test, y_pred_fused, average="macro")),
            "w_cv": float(w_cv),
            "w_glove": float(w_glove),
        },
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "classes": classes.tolist(),
    }

    report_fused = classification_report(y_test, y_pred_fused, digits=4, zero_division=0)
    return cv_model, glove_model, metrics, report_fused


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate true late fusion (CV + glove).")
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV_PATH)
    parser.add_argument("--glove", type=Path, default=DEFAULT_GLOVE_PATH)
    parser.add_argument("--pairing", choices=["label", "timestamp"], default="label")
    parser.add_argument("--tolerance-ms", type=int, default=50)
    parser.add_argument("--w-cv", type=float, default=0.6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not (0.0 <= args.w_cv <= 1.0):
        raise ValueError("--w-cv must be between 0 and 1")

    cv_small, glove_small, cv_cols, glove_cols = preprocess_streams(args.cv, args.glove, args.pairing)

    if args.pairing == "label":
        aligned = pair_by_label(cv_small, glove_small, cv_cols, glove_cols, args.random_state)
    else:
        aligned = pair_by_timestamp(cv_small, glove_small, cv_cols, glove_cols, args.tolerance_ms)

    cv_model, glove_model, metrics, report_fused = fit_and_evaluate(
        aligned, cv_cols, glove_cols, args.w_cv, args.random_state
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(cv_model, args.out_dir / "cv_model.joblib")
    joblib.dump(glove_model, args.out_dir / "glove_model.joblib")

    config = {
        "cv_path": str(args.cv),
        "glove_path": str(args.glove),
        "pairing": args.pairing,
        "tolerance_ms": args.tolerance_ms,
        "w_cv": args.w_cv,
        "w_glove": 1.0 - args.w_cv,
        "cv_features": cv_cols,
        "glove_features": glove_cols,
    }

    with open(args.out_dir / "fusion_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(args.out_dir / "fused_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_fused)

    print("Pairing mode:", args.pairing)
    print("Paired rows:", len(aligned))
    print("CV-only accuracy:", f"{metrics['cv_only']['accuracy']:.4f}")
    print("Glove-only accuracy:", f"{metrics['glove_only']['accuracy']:.4f}")
    print("Late-fused accuracy:", f"{metrics['late_fused']['accuracy']:.4f}")
    print("Saved artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()
