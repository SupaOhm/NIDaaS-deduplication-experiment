from __future__ import annotations

import time
from sklearn.metrics import precision_score, recall_score, f1_score

from src.load_clean import load_and_clean_csv, load_and_clean_folder
from src.fingerprint import build_fingerprint
from src.microbatch import iter_microbatches
from src.no_dedupe import process_no_dedupe
from src.light_detector import run_light_detector
from src.metrics import RunMetrics

def main() -> None:
    df = load_and_clean_folder("data") 
    print("Total rows:", len(df))
    print("Files loaded:", df["source_file"].nunique())
    print(df.groupby("source_file").size().sort_values(ascending=False))
    print(df["binary_label"].value_counts())
    
    df["fingerprint"] = df.apply(build_fingerprint, axis=1)

    batch_size = 5000
    metrics = RunMetrics()

    y_true_all = []
    y_pred_all = []

    for batch in iter_microbatches(df, batch_size):
        t0 = time.perf_counter()

        out_batch, stats = process_no_dedupe(batch)
        detected = run_light_detector(out_batch)

        elapsed = time.perf_counter() - t0
        metrics.add_batch(
            input_events=stats["input_events"],
            output_events=stats["output_events"],
            dropped=stats["dropped_duplicates"],
            elapsed_s=elapsed,
        )

        y_true_all.extend(detected["binary_label"].tolist())
        y_pred_all.extend(detected["pred_label"].tolist())

    summary = metrics.summary()
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    print("\n=== NoDedupe + Light Detector Baseline ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")

if __name__ == "__main__":
    main()