import argparse
from pathlib import Path

import pandas as pd

TARGET_GESTURES = ["CALL", "EAT", "HELLO", "IS", "ME", "NULL", "THANK_YOU", "YES"]


def normalize_label(value: str) -> str:
    return str(value).strip().upper().replace(" ", "_")


def extract_subset(root: Path, out_csv: Path, target_gestures: list[str]) -> None:
    targets = {normalize_label(x) for x in target_gestures}

    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {root}")

    merged = []
    per_class_rows = {k: 0 for k in sorted(targets)}
    seen_files = 0

    for csv_path in csv_files:
        rel = csv_path.relative_to(root)
        # expected layout: <date>/<gesture>/<take_dir>/<file>.csv
        folder_label = normalize_label(rel.parts[1]) if len(rel.parts) >= 2 else ""

        if folder_label not in targets:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            continue

        if "gesture" in df.columns:
            # If raw gesture is missing/empty (common for NULL class), fall back to folder label.
            g_raw = df["gesture"].fillna(folder_label).astype(str).str.strip()
            g_raw = g_raw.replace({
                "": folder_label,
                "NAN": folder_label,
                "NaN": folder_label,
                "nan": folder_label,
                "<NA>": folder_label,
                "None": folder_label,
            })
            g_norm = g_raw.map(normalize_label)
            df["gesture"] = g_norm
            df = df[df["gesture"].isin(targets)].copy()
            if df.empty:
                continue
        else:
            df["gesture"] = folder_label

        df["source_file"] = str(rel)
        df["source_label"] = folder_label

        merged.append(df)
        seen_files += 1
        counts = df["gesture"].value_counts().to_dict()
        for k, v in counts.items():
            if k in per_class_rows:
                per_class_rows[k] += int(v)

    if not merged:
        raise ValueError("No matching rows found for requested gestures.")

    out_df = pd.concat(merged, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Scanned CSV files: {len(csv_files)}")
    print(f"Used files: {seen_files}")
    print(f"Saved rows: {len(out_df)}")
    print(f"Saved file: {out_csv}")
    print("Rows per class:")
    for k in sorted(per_class_rows):
        print(f"  {k}: {per_class_rows[k]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected gesture CSV rows from raw landmark folder.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/landmark"),
        help="Root landmark directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/CV_raw_extracted.csv"),
        help="Output merged CSV path.",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=TARGET_GESTURES,
        help="Gesture labels to keep.",
    )
    args = parser.parse_args()

    extract_subset(args.root, args.out, args.gestures)


if __name__ == "__main__":
    main()
