import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("NULL")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\\s+", "_", regex=True)
    )


def detect_hand_prefixes(columns: list[str]) -> list[str]:
    prefixes = []
    for p in ("L", "R"):
        if f"{p}_x0" in columns and f"{p}_y0" in columns and f"{p}_z0" in columns:
            prefixes.append(p)
    if not prefixes:
        raise ValueError("No hand landmark columns found (L_* or R_*).")
    return prefixes


def per_hand_norm(row: pd.Series, p: str, sentinel: float, eps: float) -> tuple[list[float], int, float]:
    # Decide if hand exists using explicit flag if available, else infer from non-null wrist.
    exist_col = f"{p}_exist"
    if exist_col in row.index:
        exists = int(pd.to_numeric(row[exist_col], errors="coerce") == 1)
    else:
        exists = int(not pd.isna(row.get(f"{p}_x0", np.nan)))

    feats = []
    if not exists:
        feats.extend([sentinel] * 63)
        return feats, 0, 0.0

    wx = float(pd.to_numeric(row[f"{p}_x0"], errors="coerce"))
    wy = float(pd.to_numeric(row[f"{p}_y0"], errors="coerce"))
    wz = float(pd.to_numeric(row[f"{p}_z0"], errors="coerce"))

    x9 = float(pd.to_numeric(row[f"{p}_x9"], errors="coerce"))
    y9 = float(pd.to_numeric(row[f"{p}_y9"], errors="coerce"))
    z9 = float(pd.to_numeric(row[f"{p}_z9"], errors="coerce"))

    scale = np.sqrt((x9 - wx) ** 2 + (y9 - wy) ** 2 + (z9 - wz) ** 2)
    scale = float(max(scale, eps))

    for i in range(21):
        xi = float(pd.to_numeric(row[f"{p}_x{i}"], errors="coerce"))
        yi = float(pd.to_numeric(row[f"{p}_y{i}"], errors="coerce"))
        zi = float(pd.to_numeric(row[f"{p}_z{i}"], errors="coerce"))

        if np.isnan(xi) or np.isnan(yi) or np.isnan(zi):
            feats.extend([sentinel, sentinel, sentinel])
        else:
            feats.extend([(xi - wx) / scale, (yi - wy) / scale, (zi - wz) / scale])

    return feats, 1, scale


def build_frame_processed(df: pd.DataFrame, sentinel: float, eps: float) -> pd.DataFrame:
    prefixes = detect_hand_prefixes(list(df.columns))

    out_rows = []
    for _, row in df.iterrows():
        out = {}

        out["gesture"] = row["gesture_norm"]
        if "frame_id" in df.columns:
            out["frame_id"] = row["frame_id"]
        if "timestamp_ms" in df.columns:
            out["timestamp_ms"] = row["timestamp_ms"]
        if "source_file" in df.columns:
            out["source_file"] = row["source_file"]
        if "source_label" in df.columns:
            out["source_label"] = row["source_label"]

        for p in prefixes:
            feats, exists, scale = per_hand_norm(row, p, sentinel=sentinel, eps=eps)
            out[f"{p}_exist"] = exists
            out[f"{p}_scale"] = scale
            for i in range(21):
                out[f"{p}_xn{i}"] = feats[i * 3 + 0]
                out[f"{p}_yn{i}"] = feats[i * 3 + 1]
                out[f"{p}_zn{i}"] = feats[i * 3 + 2]

        out_rows.append(out)

    frame_df = pd.DataFrame(out_rows)
    return frame_df


def window_features(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    delta = arr[-1] - arr[0]
    if arr.shape[0] > 1:
        v = np.diff(arr, axis=0)
        v_mean = v.mean(axis=0)
        v_std = v.std(axis=0)
    else:
        v_mean = np.zeros(arr.shape[1], dtype=np.float32)
        v_std = np.zeros(arr.shape[1], dtype=np.float32)
    return np.concatenate([mean, std, delta, v_mean, v_std], axis=0)


def build_window_processed(frame_df: pd.DataFrame, window: int, step: int) -> pd.DataFrame:
    feature_cols = [c for c in frame_df.columns if c not in {"gesture", "frame_id", "timestamp_ms", "source_file", "source_label"}]

    if "source_file" in frame_df.columns:
        groups = ["source_file"]
    else:
        groups = ["gesture"]

    rows = []
    for _, g in frame_df.groupby(groups, sort=False):
        if "frame_id" in g.columns:
            g = g.sort_values("frame_id")
        elif "timestamp_ms" in g.columns:
            g = g.sort_values("timestamp_ms")

        a = g[feature_cols].to_numpy(dtype=np.float32)
        y = g["gesture"].to_numpy()

        if len(g) < window:
            continue

        for i in range(0, len(g) - window + 1, step):
            w = a[i : i + window]
            yw = y[i : i + window]
            if len(set(yw)) != 1:
                continue

            f = window_features(w)
            row = {"gesture": yw[0]}
            for j, v in enumerate(f):
                row[f"wf_{j}"] = float(v)

            if "source_file" in g.columns:
                row["source_file"] = g.iloc[i]["source_file"]
            rows.append(row)

    if not rows:
        raise ValueError("No valid windows produced. Reduce window size or check sequence grouping.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CV preprocessing pipeline: frame + temporal window features.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/CV_raw_extracted.csv"))
    parser.add_argument("--frame-out", type=Path, default=Path("data/processed/cv_frame_processed.csv"))
    parser.add_argument("--window-out", type=Path, default=Path("data/processed/cv_window_processed.csv"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--exclude-null", action="store_true")
    parser.add_argument("--sentinel", type=float, default=-10.0, help="Value used when a hand is missing.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Minimum scale denominator.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    if "gesture" not in df.columns:
        raise ValueError("Input CSV must contain 'gesture' column.")

    df["gesture_norm"] = normalize_label(df["gesture"])
    if args.exclude_null:
        df = df[df["gesture_norm"] != "NULL"].copy()

    frame_df = build_frame_processed(df, sentinel=args.sentinel, eps=args.eps)
    window_df = build_window_processed(frame_df, window=args.window, step=args.step)

    args.frame_out.parent.mkdir(parents=True, exist_ok=True)
    args.window_out.parent.mkdir(parents=True, exist_ok=True)

    frame_df.to_csv(args.frame_out, index=False)
    window_df.to_csv(args.window_out, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Frame processed rows: {len(frame_df)}, cols: {len(frame_df.columns)}")
    print(f"Window processed rows: {len(window_df)}, cols: {len(window_df.columns)}")
    print(f"Saved: {args.frame_out}")
    print(f"Saved: {args.window_out}")


if __name__ == "__main__":
    main()
