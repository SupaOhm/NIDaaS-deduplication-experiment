from __future__ import annotations

import time
from sklearn.metrics import precision_score, recall_score, f1_score

from src.load_clean import load_and_clean_folder
from src.fingerprint import build_fingerprint
from src.microbatch import iter_microbatches
from src.ids.rf_detector import RFDetector
from src.metrics import RunMetrics
from src.exact_map import ExactMapDedupe


def main() -> None:
    df = load_and_clean_folder("data")
    print("Total rows:", len(df))
    print("Files loaded:", df["source_file"].nunique())
    print(df.groupby("source_file").size().sort_values(ascending=False))
    print(df["binary_label"].value_counts())

    df["fingerprint"] = df.apply(build_fingerprint, axis=1)

    batch_size = 5000
    metrics = RunMetrics()
    detector = RFDetector()
    deduper = ExactMapDedupe(max_recent=50000)

    y_true_all = []
    y_pred_all = []

    dropped_benign = 0
    dropped_attack = 0

    for batch in iter_microbatches(df, batch_size):
        t0 = time.perf_counter()

        out_batch, dropped_batch, stats = deduper.process_batch(batch)
        detected = detector.predict_batch(out_batch)

        elapsed = time.perf_counter() - t0
        metrics.add_batch(
            input_events=stats["input_events"],
            output_events=stats["output_events"],
            dropped=stats["dropped_duplicates"],
            elapsed_s=elapsed,
        )

        dropped_benign += (dropped_batch["binary_label"] == 0).sum()
        dropped_attack += (dropped_batch["binary_label"] == 1).sum()

        y_true_all.extend(detected["binary_label"].tolist())
        y_pred_all.extend(detected["pred_label"].tolist())

    summary = metrics.summary()
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    print("\n=== ExactMap + RF Baseline ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"final_state_size: {len(deduper.seen)}")
    print(f"dropped_benign: {dropped_benign}")
    print(f"dropped_attack: {dropped_attack}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")


if __name__ == "__main__":
    main()