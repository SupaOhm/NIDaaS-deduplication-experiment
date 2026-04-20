from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.load_clean import load_and_clean_folder
from src.fingerprint import (
    build_fingerprint_basic,
    build_fingerprint_with_duration,
    build_fingerprint_with_duration_bucket,
    build_fingerprint_with_packet_counts,
)
from src.microbatch import iter_microbatches
from src.ids.rf_detector import RFDetector
from src.metrics import RunMetrics
from src.exact_map import ExactMapDedupe


FINGERPRINT_BUILDERS: dict[str, Callable[[pd.Series], str]] = {
    "basic": build_fingerprint_basic,
    "duration_bucket": build_fingerprint_with_duration_bucket,
    "packet_counts": build_fingerprint_with_packet_counts,
    "duration": build_fingerprint_with_duration,
}
DEFAULT_FINGERPRINT_MODE = "packet_counts"
FINGERPRINT_MODE_GROUPS = {
    "ab": ["basic", "duration"],
    "all": ["basic", "duration_bucket", DEFAULT_FINGERPRINT_MODE, "duration"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ExactMap + RF dedupe experiment.")
    parser.add_argument(
        "--fingerprint-mode",
        choices=[*FINGERPRINT_BUILDERS, *FINGERPRINT_MODE_GROUPS],
        default=DEFAULT_FINGERPRINT_MODE,
        help=(
            f"Fingerprint variant to test. Default is {DEFAULT_FINGERPRINT_MODE}. "
            "Use 'ab' for basic vs duration, or 'all' for all variants."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-recent", type=int, default=50000)
    return parser.parse_args()


def selected_fingerprint_modes(mode: str) -> list[str]:
    return FINGERPRINT_MODE_GROUPS.get(mode, [mode])


def add_fingerprints(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["fingerprint"] = out.apply(FINGERPRINT_BUILDERS[mode], axis=1)
    return out


def run_experiment(df: pd.DataFrame, mode: str, batch_size: int, max_recent: int) -> dict:
    print(f"\nFingerprint mode: {mode.upper()}")

    run_df = add_fingerprints(df, mode)
    metrics = RunMetrics()
    detector = RFDetector()
    deduper = ExactMapDedupe(max_recent=max_recent)

    y_true_all = []
    y_pred_all = []

    dropped_benign = 0
    dropped_attack = 0

    for batch in iter_microbatches(run_df, batch_size):
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
    drop_rate = (
        summary["total_dropped"] / summary["total_input"]
        if summary["total_input"]
        else 0.0
    )

    result = {
        **summary,
        "fingerprint_mode": mode,
        "drop_rate": drop_rate,
        "final_state_size": len(deduper.seen),
        "dropped_benign": dropped_benign,
        "dropped_attack": dropped_attack,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print("\n=== ExactMap + RF Baseline ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"final_state_size: {result['final_state_size']}")
    print(f"dropped_benign: {dropped_benign}")
    print(f"dropped_attack: {dropped_attack}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")

    return result


def print_comparison_table(results: list[dict]) -> None:
    if len(results) < 2:
        return

    print("\n=== Fingerprint Comparison ===")
    print(
        f"{'mode':<16} {'dropped':>10} {'drop_rate':>10} "
        f"{'drop_attack':>12} {'drop_benign':>12} {'precision':>10} {'recall':>8} {'f1':>8}"
    )
    for result in results:
        print(
            f"{result['fingerprint_mode']:<16} "
            f"{result['total_dropped']:>10} "
            f"{result['drop_rate']:>9.4%} "
            f"{result['dropped_attack']:>12} "
            f"{result['dropped_benign']:>12} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>8.4f} "
            f"{result['f1']:>8.4f}"
        )


def print_ab_delta(results: list[dict]) -> None:
    if len(results) != 2:
        return

    basic, duration = results
    print("\n=== Fingerprint A/B Delta (duration - basic) ===")
    for key in ["total_dropped", "dropped_benign", "dropped_attack", "final_state_size"]:
        print(f"{key}: {duration[key] - basic[key]}")
    print(f"precision: {duration['precision'] - basic['precision']:.4f}")
    print(f"recall:    {duration['recall'] - basic['recall']:.4f}")
    print(f"f1:        {duration['f1'] - basic['f1']:.4f}")


def main() -> None:
    args = parse_args()

    df = load_and_clean_folder("data")
    print("Total rows:", len(df))
    print("Files loaded:", df["source_file"].nunique())
    print(df.groupby("source_file").size().sort_values(ascending=False))
    print(df["binary_label"].value_counts())

    modes = selected_fingerprint_modes(args.fingerprint_mode)
    results = [
        run_experiment(df, mode, args.batch_size, args.max_recent)
        for mode in modes
    ]
    print_comparison_table(results)
    print_ab_delta(results)


if __name__ == "__main__":
    main()
