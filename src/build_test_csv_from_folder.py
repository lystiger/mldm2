import argparse
from pathlib import Path

import pandas as pd


def normalize_label(value: str) -> str:
    return str(value).strip().upper().replace(" ", "_")


def infer_label_from_filename(path: Path) -> str:
    # expected pattern: landmarks_<Gesture>_...csv
    stem = path.stem
    if stem.lower().startswith("landmarks_"):
        rest = stem[len("landmarks_") :]
        # gesture part is token(s) before date chunk
        # we take first chunk up to year-like pattern as fallback
        parts = rest.split("_")
        if parts:
            # handle multi-word gesture e.g. "Thank You"
            # collect tokens until one looks like date (contains '-')
            label_tokens = []
            for p in parts:
                if "-" in p:
                    break
                label_tokens.append(p)
            if label_tokens:
                return normalize_label(" ".join(label_tokens))
    return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all raw test CSV files from a folder into one CSV.")
    parser.add_argument("--root", type=Path, default=Path("data/test/cv_raw"))
    parser.add_argument("--out", type=Path, default=Path("data/test/cv_raw/CV_test_raw_merged.csv"))
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Folder not found: {args.root}")

    files = sorted([p for p in args.root.glob("*.csv") if p.name != args.out.name])
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {args.root}")

    chunks = []
    used = 0
    for p in files:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        inferred = infer_label_from_filename(p)

        if "gesture" in df.columns:
            g = df["gesture"].fillna("").astype(str).str.strip()
            g = g.replace({"": inferred})
            df["gesture"] = g.map(normalize_label)
        else:
            df["gesture"] = inferred

        df["source_file"] = p.name
        chunks.append(df)
        used += 1

    if not chunks:
        raise ValueError("Could not read any valid CSV rows.")

    out_df = pd.concat(chunks, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Input files found: {len(files)}")
    print(f"Files merged: {used}")
    print(f"Rows written: {len(out_df)}")
    print(f"Saved: {args.out}")
    if "gesture" in out_df.columns:
        print("Label counts:")
        vc = out_df["gesture"].value_counts(dropna=False)
        for k, v in vc.items():
            print(f"  {k}: {int(v)}")


if __name__ == "__main__":
    main()
