from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.duplicate_injector import ALLOWED_DUPLICATE_RATES, inject_exact_replays
from src.exact_map import ExactMapDedupe
from src.fingerprint import build_fingerprint_with_packet_counts
from src.ids.rf_detector import RFDetector
from src.load_clean import load_and_clean_folder
from src.microbatch import iter_microbatches
from src.no_dedupe import NoDedupe


DEFAULT_DUPLICATE_RATES = [0, 10, 20]
DEFAULT_RF_COST_MULTIPLIERS = [1, 2, 5, 10]
DEDUPE_MODES = ["no_dedupe", "exact_map"]
TABLE_COLUMNS = [
    "dedupe_mode",
    "duplicate_rate",
    "rf_cost_multiplier",
    "total_input",
    "total_output",
    "dropped",
    "dropped_attack",
    "dropped_benign",
    "dedupe_time",
    "rf_time",
    "total_time",
    "throughput",
    "precision",
    "recall",
    "f1",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def duplicate_rate(value: str) -> int:
    parsed = int(value)
    if parsed not in ALLOWED_DUPLICATE_RATES:
        raise argparse.ArgumentTypeError(
            f"value must be one of {ALLOWED_DUPLICATE_RATES}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Side experiment for when packet_counts deduplication becomes "
            "worthwhile before fixed RF inference."
        )
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=positive_int, default=5000)
    parser.add_argument("--max-recent", type=positive_int, default=50000)
    parser.add_argument(
        "--max-rows",
        type=positive_int,
        default=None,
        help="Optional row cap applied after cleaning and before duplicate injection.",
    )
    parser.add_argument(
        "--sample-rows",
        type=positive_int,
        default=None,
        help=(
            "Optional order-preserving random sample applied after cleaning "
            "and before duplicate injection."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--duplicate-rates",
        "--duplicate-rate",
        dest="duplicate_rates",
        type=duplicate_rate,
        nargs="+",
        default=DEFAULT_DUPLICATE_RATES,
        help="Exact replay injection rates as percentages of original rows.",
    )
    parser.add_argument("--duplicate-seed", type=int, default=42)
    parser.add_argument(
        "--rf-cost-multipliers",
        "--rf-cost-multiplier",
        dest="rf_cost_multipliers",
        type=positive_int,
        nargs="+",
        default=DEFAULT_RF_COST_MULTIPLIERS,
        help="Repeat RF prediction N times on the same surviving batch.",
    )
    parser.add_argument(
        "--dedupe-modes",
        "--dedupe-mode",
        dest="dedupe_modes",
        choices=DEDUPE_MODES,
        nargs="+",
        default=DEDUPE_MODES,
    )
    return parser.parse_args()


def add_packet_count_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fingerprint"] = out.apply(build_fingerprint_with_packet_counts, axis=1)
    return out


def apply_max_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(df):
        return df.copy()
    return df.head(max_rows).copy()


def apply_sample_rows(
    df: pd.DataFrame,
    sample_rows: int | None,
    sample_seed: int,
) -> pd.DataFrame:
    if sample_rows is None or sample_rows >= len(df):
        return df.copy()
    return (
        df.sample(n=sample_rows, random_state=sample_seed)
        .sort_index(kind="mergesort")
        .reset_index(drop=True)
    )


def make_deduper(mode: str, max_recent: int):
    if mode == "no_dedupe":
        return NoDedupe()
    if mode == "exact_map":
        return ExactMapDedupe(max_recent=max_recent)
    raise ValueError(f"Unknown dedupe mode: {mode}")


def predict_with_multiplier(
    detector: RFDetector,
    batch: pd.DataFrame,
    rf_cost_multiplier: int,
) -> tuple[pd.DataFrame, float]:
    if batch.empty:
        detected = batch.copy()
        detected["pred_label"] = pd.Series(index=detected.index, dtype="int64")
        return detected, 0.0

    first_detected: pd.DataFrame | None = None
    t0 = time.perf_counter()
    for _ in range(rf_cost_multiplier):
        detected = detector.predict_batch(batch)
        if first_detected is None:
            first_detected = detected

    rf_time = time.perf_counter() - t0
    if first_detected is None:
        raise RuntimeError("RF prediction did not run")
    return first_detected, rf_time


def run_one(
    df: pd.DataFrame,
    detector: RFDetector,
    dedupe_mode: str,
    duplicate_rate_value: int,
    rf_cost_multiplier: int,
    batch_size: int,
    max_recent: int,
) -> dict:
    deduper = make_deduper(dedupe_mode, max_recent=max_recent)

    total_input = 0
    total_output = 0
    dropped = 0
    dropped_attack = 0
    dropped_benign = 0
    dedupe_time = 0.0
    rf_time = 0.0
    total_time = 0.0
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for batch in iter_microbatches(df, batch_size):
        batch_t0 = time.perf_counter()

        dedupe_t0 = time.perf_counter()
        out_batch, dropped_batch, stats = deduper.process_batch(batch)
        dedupe_elapsed = time.perf_counter() - dedupe_t0

        detected, rf_elapsed = predict_with_multiplier(
            detector,
            out_batch,
            rf_cost_multiplier=rf_cost_multiplier,
        )
        batch_elapsed = time.perf_counter() - batch_t0

        total_input += stats["input_events"]
        total_output += stats["output_events"]
        dropped += stats["dropped_duplicates"]
        dropped_attack += int((dropped_batch["binary_label"] == 1).sum())
        dropped_benign += int((dropped_batch["binary_label"] == 0).sum())
        dedupe_time += dedupe_elapsed
        rf_time += rf_elapsed
        total_time += batch_elapsed

        y_true_all.extend(detected["binary_label"].tolist())
        y_pred_all.extend(detected["pred_label"].tolist())

    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    return {
        "dedupe_mode": dedupe_mode,
        "duplicate_rate": duplicate_rate_value,
        "rf_cost_multiplier": rf_cost_multiplier,
        "total_input": total_input,
        "total_output": total_output,
        "dropped": dropped,
        "dropped_attack": dropped_attack,
        "dropped_benign": dropped_benign,
        "dedupe_time": dedupe_time,
        "rf_time": rf_time,
        "total_time": total_time,
        "throughput": total_input / total_time if total_time > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_result_table(results: list[dict]) -> None:
    table = pd.DataFrame(results, columns=TABLE_COLUMNS)
    formatters = {
        "dedupe_time": "{:.6f}".format,
        "rf_time": "{:.6f}".format,
        "total_time": "{:.6f}".format,
        "throughput": "{:.0f}".format,
        "precision": "{:.4f}".format,
        "recall": "{:.4f}".format,
        "f1": "{:.4f}".format,
    }
    print("\n=== RF Cost Sensitivity Comparison ===")
    print(table.to_string(index=False, formatters=formatters))


def print_timing_delta_table(results: list[dict]) -> None:
    table = pd.DataFrame(results)
    if not {"no_dedupe", "exact_map"}.issubset(set(table["dedupe_mode"])):
        return

    rows = []
    for (rate, multiplier), group in table.groupby(
        ["duplicate_rate", "rf_cost_multiplier"],
        sort=True,
    ):
        by_mode = group.set_index("dedupe_mode")
        if "no_dedupe" not in by_mode.index or "exact_map" not in by_mode.index:
            continue

        baseline = by_mode.loc["no_dedupe"]
        exact = by_mode.loc["exact_map"]
        rows.append(
            {
                "duplicate_rate": rate,
                "rf_cost_multiplier": multiplier,
                "dropped": int(exact["dropped"]),
                "dedupe_time_added": exact["dedupe_time"] - baseline["dedupe_time"],
                "rf_time_saved": baseline["rf_time"] - exact["rf_time"],
                "total_time_saved": baseline["total_time"] - exact["total_time"],
                "throughput_gain": exact["throughput"] - baseline["throughput"],
                "speedup": (
                    baseline["total_time"] / exact["total_time"]
                    if exact["total_time"] > 0
                    else 0.0
                ),
            }
        )

    if not rows:
        return

    delta = pd.DataFrame(rows)
    formatters = {
        "dedupe_time_added": "{:.6f}".format,
        "rf_time_saved": "{:.6f}".format,
        "total_time_saved": "{:.6f}".format,
        "throughput_gain": "{:.0f}".format,
        "speedup": "{:.3f}".format,
    }
    print("\n=== ExactMap Timing Delta vs NoDedupe ===")
    print(delta.to_string(index=False, formatters=formatters))


def main() -> None:
    args = parse_args()

    df = load_and_clean_folder(args.data_dir)
    print("Total rows:", len(df))
    print("Files loaded:", df["source_file"].nunique())
    print(df["binary_label"].value_counts())

    loaded_rows = len(df)
    df = apply_max_rows(df, args.max_rows)
    if args.max_rows is not None:
        print("\nRow limit:")
        print(f"max_rows: {args.max_rows}")
        print(f"loaded_rows: {loaded_rows}")
        print(f"rows_after_limit: {len(df)}")

    capped_rows = len(df)
    df = apply_sample_rows(df, args.sample_rows, args.sample_seed)
    if args.sample_rows is not None:
        print("\nRow sample:")
        print(f"sample_rows: {args.sample_rows}")
        print(f"sample_seed: {args.sample_seed}")
        print(f"rows_before_sample: {capped_rows}")
        print(f"rows_after_sample: {len(df)}")

    print("\nFingerprint mode: PACKET_COUNTS")
    detector = RFDetector()
    results = []

    for rate in args.duplicate_rates:
        injected_df, injection_stats = inject_exact_replays(
            df,
            duplicate_rate=rate,
            random_seed=args.duplicate_seed,
        )
        print("\nDuplicate injection:")
        print(f"mode: {injection_stats['injection_mode']}")
        print(f"duplicate_rate: {injection_stats['duplicate_rate']}%")
        print(f"random_seed: {injection_stats['random_seed']}")
        print(f"original_rows: {injection_stats['original_rows']}")
        print(f"injected_rows: {injection_stats['injected_rows']}")
        print(f"total_rows_after_injection: {injection_stats['total_rows']}")

        run_df = add_packet_count_fingerprints(injected_df)
        for multiplier in args.rf_cost_multipliers:
            for mode in args.dedupe_modes:
                results.append(
                    run_one(
                        run_df,
                        detector,
                        dedupe_mode=mode,
                        duplicate_rate_value=rate,
                        rf_cost_multiplier=multiplier,
                        batch_size=args.batch_size,
                        max_recent=args.max_recent,
                    )
                )

    print_result_table(results)
    print_timing_delta_table(results)


if __name__ == "__main__":
    main()
