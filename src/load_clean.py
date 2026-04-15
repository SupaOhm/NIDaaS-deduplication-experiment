from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

KEEP_COLUMNS = [
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "Average Packet Size",
    "Label",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def load_and_clean_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[warn] utf-8 failed for {path}, retrying with latin1")
        df = pd.read_csv(path, low_memory=False, encoding="latin1")

    df = normalize_column_names(df)

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df[KEEP_COLUMNS].copy()

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Timestamp_rounded_1s"] = df["Timestamp"].dt.floor("1s")

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with broken core identity fields
    core_identity = [
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        "Timestamp",
    ]
    df = df.dropna(subset=core_identity)

    # Clean numeric columns
    numeric_cols = [
        "Source Port",
        "Destination Port",
        "Protocol",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "SYN Flag Count",
        "ACK Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "Average Packet Size",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    important_numeric = [
        "Source Port",
        "Destination Port",
        "Protocol",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
    ]
    df = df.dropna(subset=important_numeric)

    fill_zero_cols = [
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "SYN Flag Count",
        "ACK Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "Average Packet Size",
    ]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    df["Label"] = df["Label"].astype(str).str.strip()
    df["binary_label"] = (df["Label"].str.upper() != "BENIGN").astype(int)

    df = df.reset_index(drop=True)
    return df

def load_and_clean_folder(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in: {folder_path}")

    frames = []
    for csv_file in csv_files:
        print(f"[load] {csv_file.name}")
        try:
            df = load_and_clean_csv(str(csv_file)).assign(source_file=csv_file.name)
            print(f"  rows after cleaning: {len(df)}")
            frames.append(df)
        except Exception as e:
            print(f"[error] failed on {csv_file.name}: {e}")
            raise

    merged = pd.concat(frames, ignore_index=True)
    return merged

if __name__ == "__main__":
    path = "data"
    df = load_and_clean_folder(path)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head())
    print(df["binary_label"].value_counts(dropna=False))