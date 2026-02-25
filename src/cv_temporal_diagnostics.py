import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_DATASET = Path("data/raw/CV_raw_extracted.csv")
DEFAULT_OUT_DIR = Path("data/models/cv_temporal_diagnostics")


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("NULL")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def get_landmark_cols(columns):
    r_cols = [c for c in columns if c.startswith("R_") and len(c) >= 4 and c[2] in ("x", "y", "z")]
    l_cols = [c for c in columns if c.startswith("L_") and len(c) >= 4 and c[2] in ("x", "y", "z")]
    if r_cols:
        return sorted(r_cols), "R"
    if l_cols:
        return sorted(l_cols), "L"
    raise ValueError("No landmark columns found.")


def feature_vector_from_window(arr: np.ndarray) -> np.ndarray:
    # arr shape: [window_size, n_features]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    delta = arr[-1] - arr[0]

    # velocity-like features from frame differences
    if arr.shape[0] > 1:
        vel = np.diff(arr, axis=0)
        vel_mean = vel.mean(axis=0)
        vel_std = vel.std(axis=0)
    else:
        vel_mean = np.zeros(arr.shape[1], dtype=np.float32)
        vel_std = np.zeros(arr.shape[1], dtype=np.float32)

    return np.concatenate([mean, std, delta, vel_mean, vel_std], axis=0)


def build_windows(df: pd.DataFrame, feat_cols: list[str], window: int, step: int):
    if "source_file" in df.columns:
        group_keys = ["source_file"]
    else:
        group_keys = ["gesture"]

    X, y = [], []

    for _, g in df.groupby(group_keys, sort=False):
        if "frame_id" in g.columns:
            g = g.sort_values("frame_id")
        elif "timestamp_ms" in g.columns:
            g = g.sort_values("timestamp_ms")

        a = g[feat_cols].to_numpy(dtype=np.float32)
        labels = g["gesture_norm"].to_numpy()

        if len(g) < window:
            continue

        for i in range(0, len(g) - window + 1, step):
            w = a[i : i + window]
            lbl = labels[i : i + window]
            # keep only clean windows with consistent label
            if len(set(lbl)) != 1:
                continue
            X.append(feature_vector_from_window(w))
            y.append(lbl[0])

    if not X:
        raise ValueError("No valid windows built. Reduce window size or check data ordering/labels.")

    return np.vstack(X), np.array(y)


def top_confusions(cm: np.ndarray, classes: list[str], k: int = 12):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), classes[i], classes[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:k]


def main():
    parser = argparse.ArgumentParser(description="Temporal CV diagnostics with window features.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--exclude-null", action="store_true")
    parser.add_argument("--model", choices=["svm", "rf", "knn", "logreg"], default="svm")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data, low_memory=False)
    if "gesture" not in df.columns:
        raise ValueError("Dataset must contain 'gesture' column.")

    df["gesture_norm"] = normalize_label(df["gesture"])
    feat_cols, hand_mode = get_landmark_cols(df.columns)
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if args.exclude_null:
        df = df[df["gesture_norm"] != "NULL"].copy()

    X, y = build_windows(df, feat_cols, args.window, args.step)

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
    conf_pairs = top_confusions(cm, classes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = args.out_dir / "summary.txt"

    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Hand mode: {hand_mode}\n")
        f.write(f"Landmark dims per frame: {len(feat_cols)}\n")
        f.write(f"Window size: {args.window} | Step: {args.step}\n")
        f.write(f"Window feature dims: {X.shape[1]}\n")
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

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_title(f"Temporal Confusion Matrix ({args.model.upper()})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"Window samples: {len(X)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Saved: {summary}")
    print(f"Saved: {args.out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
