from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.bloom import BloomDedupe
from src.bloom_exact import BloomExactDedupe
from src.duplicate_injector import ALLOWED_DUPLICATE_RATES, inject_exact_replays
from src.exact_map import ExactMapDedupe
from src.fingerprint import build_fingerprint_with_packet_counts
from src.ids.rf_detector import RFDetector
from src.load_clean import load_and_clean_folder
from src.metrics import RunMetrics
from src.microbatch import iter_microbatches


DEFAULT_BLOOM_BITS = [10_000_000, 25_000_000, 50_000_000, 100_000_000]
DEFAULT_BLOOM_HASHES = [2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side experiment for Bloom fairness under fixed packet_counts fingerprints."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-recent", type=int, default=50000)
    parser.add_argument(
        "--duplicate-rate",
        type=int,
        choices=ALLOWED_DUPLICATE_RATES,
        default=0,
        help="Exact replay injection rate as a percent of original rows.",
    )
    parser.add_argument("--duplicate-seed", type=int, default=42)
    parser.add_argument("--bloom-bits", type=int, nargs="+", default=DEFAULT_BLOOM_BITS)
    parser.add_argument(
        "--bloom-hashes",
        type=int,
        nargs="+",
        default=DEFAULT_BLOOM_HASHES,
    )
    return parser.parse_args()


def add_packet_count_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fingerprint"] = out.apply(build_fingerprint_with_packet_counts, axis=1)
    return out


def make_bloom_bytes(bit_size: int | None) -> int | None:
    if bit_size is None:
        return None
    return (bit_size + 7) // 8


def run_one(
    df: pd.DataFrame,
    detector: RFDetector,
    deduper,
    dedupe_mode: str,
    batch_size: int,
    bloom_bits: int | None = None,
    bloom_hashes: int | None = None,
) -> dict:
    metrics = RunMetrics()
    last_stats: dict = {"state_size": 0}

    y_true_all = []
    y_pred_all = []
    dropped_benign = 0
    dropped_attack = 0

    for batch in iter_microbatches(df, batch_size):
        t0 = time.perf_counter()

        out_batch, dropped_batch, stats = deduper.process_batch(batch)
        last_stats = stats
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
    return {
        "dedupe_mode": dedupe_mode,
        "bloom_bits": bloom_bits,
        "bloom_hashes": bloom_hashes,
        "bloom_bytes": make_bloom_bytes(bloom_bits),
        "dropped": summary["total_dropped"],
        "dropped_attack": dropped_attack,
        "dropped_benign": dropped_benign,
        "throughput": summary["throughput_events_per_s"],
        "final_state_size": last_stats["state_size"],
        "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
        "recall": recall_score(y_true_all, y_pred_all, zero_division=0),
        "f1": f1_score(y_true_all, y_pred_all, zero_division=0),
    }


def print_result_table(results: list[dict]) -> None:
    print("\n=== Bloom Fairness Comparison ===")
    print(
        f"{'dedupe':<12} {'bits':>10} {'hashes':>6} {'dropped':>10} "
        f"{'drop_attack':>12} {'drop_benign':>12} {'throughput':>12} "
        f"{'state':>10} {'bloom_bytes':>12} {'precision':>10} {'recall':>8} {'f1':>8}"
    )
    for result in results:
        bits = "-" if result["bloom_bits"] is None else str(result["bloom_bits"])
        hashes = "-" if result["bloom_hashes"] is None else str(result["bloom_hashes"])
        bloom_bytes = (
            "-"
            if result["bloom_bytes"] is None
            else str(result["bloom_bytes"])
        )
        print(
            f"{result['dedupe_mode']:<12} "
            f"{bits:>10} "
            f"{hashes:>6} "
            f"{result['dropped']:>10} "
            f"{result['dropped_attack']:>12} "
            f"{result['dropped_benign']:>12} "
            f"{result['throughput']:>12.0f} "
            f"{result['final_state_size']:>10} "
            f"{bloom_bytes:>12} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>8.4f} "
            f"{result['f1']:>8.4f}"
        )


def main() -> None:
    args = parse_args()

    df = load_and_clean_folder(args.data_dir)
    print("Total rows:", len(df))
    print("Files loaded:", df["source_file"].nunique())
    print(df["binary_label"].value_counts())

    df, injection_stats = inject_exact_replays(
        df,
        duplicate_rate=args.duplicate_rate,
        random_seed=args.duplicate_seed,
    )
    print("\nDuplicate injection:")
    print(f"mode: {injection_stats['injection_mode']}")
    print(f"duplicate_rate: {injection_stats['duplicate_rate']}%")
    print(f"random_seed: {injection_stats['random_seed']}")
    print(f"original_rows: {injection_stats['original_rows']}")
    print(f"injected_rows: {injection_stats['injected_rows']}")
    print(f"total_rows_after_injection: {injection_stats['total_rows']}")

    print("\nFingerprint mode: PACKET_COUNTS")
    run_df = add_packet_count_fingerprints(df)
    detector = RFDetector()

    results = [
        run_one(
            run_df,
            detector,
            ExactMapDedupe(max_recent=args.max_recent),
            "exact_map",
            args.batch_size,
        )
    ]

    for bit_size in args.bloom_bits:
        for num_hashes in args.bloom_hashes:
            results.append(
                run_one(
                    run_df,
                    detector,
                    BloomDedupe(bit_size=bit_size, num_hashes=num_hashes),
                    "bloom",
                    args.batch_size,
                    bloom_bits=bit_size,
                    bloom_hashes=num_hashes,
                )
            )
            results.append(
                run_one(
                    run_df,
                    detector,
                    BloomExactDedupe(
                        bit_size=bit_size,
                        num_hashes=num_hashes,
                        max_recent=args.max_recent,
                    ),
                    "bloom_exact",
                    args.batch_size,
                    bloom_bits=bit_size,
                    bloom_hashes=num_hashes,
                )
            )

    print_result_table(results)


if __name__ == "__main__":
    main()
