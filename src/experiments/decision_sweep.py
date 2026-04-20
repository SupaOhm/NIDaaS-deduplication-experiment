from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.bloom import BloomDedupe
from src.bloom_exact import BloomExactDedupe
from src.duplicate_injector import inject_exact_replays
from src.exact_map import ExactMapDedupe
from src.fingerprint import build_fingerprint_with_packet_counts
from src.ids.rf_detector import RFDetector
from src.load_clean import load_and_clean_folder
from src.microbatch import iter_microbatches
from src.no_dedupe import NoDedupe


DUPLICATE_RATES = [0, 5, 10, 20, 30, 40, 50]
DEDUPE_MODES = ["no_dedupe", "exact_map", "bloom", "bloom_exact"]
DEDUPE_ENABLED_MODES = ["exact_map", "bloom", "bloom_exact"]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def duplicate_rate(value: str) -> int:
    parsed = int(value)
    if parsed not in DUPLICATE_RATES:
        raise argparse.ArgumentTypeError(f"value must be one of {DUPLICATE_RATES}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-branch-only decision sweep for dedupe runtime tradeoffs."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=positive_int, default=5000)
    parser.add_argument("--max-recent", type=positive_int, default=50000)
    parser.add_argument("--bloom-bits", type=positive_int, default=50_000_000)
    parser.add_argument("--bloom-hashes", type=positive_int, default=4)
    parser.add_argument(
        "--max-rows",
        type=positive_int,
        default=None,
        help="Optional row cap applied after cleaning and before duplicate injection.",
    )
    parser.add_argument(
        "--duplicate-rates",
        type=duplicate_rate,
        nargs="+",
        default=DUPLICATE_RATES,
        help="Exact replay duplicate rates to sweep.",
    )
    parser.add_argument("--duplicate-seed", type=int, default=42)
    return parser.parse_args()


def apply_max_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(df):
        return df.copy()
    return df.head(max_rows).copy()


def add_packet_count_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fingerprint"] = out.apply(build_fingerprint_with_packet_counts, axis=1)
    return out


def make_deduper(mode: str, args: argparse.Namespace):
    if mode == "no_dedupe":
        return NoDedupe()
    if mode == "exact_map":
        return ExactMapDedupe(max_recent=args.max_recent)
    if mode == "bloom":
        return BloomDedupe(
            bit_size=args.bloom_bits,
            num_hashes=args.bloom_hashes,
        )
    if mode == "bloom_exact":
        return BloomExactDedupe(
            bit_size=args.bloom_bits,
            num_hashes=args.bloom_hashes,
            max_recent=args.max_recent,
        )
    raise ValueError(f"Unknown dedupe mode: {mode}")


def predict_batch(detector: RFDetector, batch: pd.DataFrame) -> pd.DataFrame:
    if not batch.empty:
        return detector.predict_batch(batch)

    out = batch.copy()
    out["pred_label"] = pd.Series(index=out.index, dtype="int64")
    return out


def run_one(
    df: pd.DataFrame,
    detector: RFDetector,
    dedupe_mode: str,
    duplicate_rate_value: int,
    args: argparse.Namespace,
) -> dict:
    deduper = make_deduper(dedupe_mode, args)

    total_input = 0
    total_output = 0
    dropped = 0
    dropped_attack = 0
    dropped_benign = 0
    dedupe_time = 0.0
    rf_time = 0.0
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for batch in iter_microbatches(df, args.batch_size):
        dedupe_t0 = time.perf_counter()
        out_batch, dropped_batch, stats = deduper.process_batch(batch)
        dedupe_time += time.perf_counter() - dedupe_t0

        rf_t0 = time.perf_counter()
        detected = predict_batch(detector, out_batch)
        rf_time += time.perf_counter() - rf_t0

        total_input += stats["input_events"]
        total_output += stats["output_events"]
        dropped += stats["dropped_duplicates"]
        dropped_attack += int((dropped_batch["binary_label"] == 1).sum())
        dropped_benign += int((dropped_batch["binary_label"] == 0).sum())

        y_true_all.extend(detected["binary_label"].tolist())
        y_pred_all.extend(detected["pred_label"].tolist())

    total_time = dedupe_time + rf_time
    return {
        "duplicate_rate": duplicate_rate_value,
        "dedupe_mode": dedupe_mode,
        "total_input": total_input,
        "total_output": total_output,
        "dropped": dropped,
        "dropped_attack": dropped_attack,
        "dropped_benign": dropped_benign,
        "dedupe_time": dedupe_time,
        "rf_time": rf_time,
        "total_time": total_time,
        "throughput": total_input / total_time if total_time > 0 else 0.0,
        "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
        "recall": recall_score(y_true_all, y_pred_all, zero_division=0),
        "f1": f1_score(y_true_all, y_pred_all, zero_division=0),
    }


def add_no_dedupe_deltas(results: list[dict]) -> list[dict]:
    by_rate_mode = {
        (row["duplicate_rate"], row["dedupe_mode"]): row
        for row in results
    }

    rows = []
    for row in results:
        baseline = by_rate_mode[(row["duplicate_rate"], "no_dedupe")]
        total_time_saved = baseline["total_time"] - row["total_time"]
        speedup = (
            baseline["total_time"] / row["total_time"]
            if row["total_time"] > 0
            else 0.0
        )
        rows.append({
            **row,
            "total_time_saved_vs_no_dedupe": total_time_saved,
            "speedup_vs_no_dedupe": speedup,
        })
    return rows


def print_result_table(results: list[dict]) -> None:
    table = pd.DataFrame(results)
    columns = [
        "duplicate_rate",
        "dedupe_mode",
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
        "total_time_saved_vs_no_dedupe",
        "speedup_vs_no_dedupe",
    ]
    formatters = {
        "dedupe_time": "{:.6f}".format,
        "rf_time": "{:.6f}".format,
        "total_time": "{:.6f}".format,
        "throughput": "{:.0f}".format,
        "precision": "{:.4f}".format,
        "recall": "{:.4f}".format,
        "f1": "{:.4f}".format,
        "total_time_saved_vs_no_dedupe": "{:.6f}".format,
        "speedup_vs_no_dedupe": "{:.3f}".format,
    }
    print("\n=== RF Branch Decision Sweep ===")
    print(table[columns].to_string(index=False, formatters=formatters))


def first_exact_map_crossover(results: list[dict]) -> int | None:
    for rate in sorted({row["duplicate_rate"] for row in results}):
        exact = next(
            row for row in results
            if row["duplicate_rate"] == rate and row["dedupe_mode"] == "exact_map"
        )
        if exact["total_time_saved_vs_no_dedupe"] > 0:
            return rate
    return None


def best_mode_by_mean_time(results: list[dict], modes: list[str]) -> tuple[str, float]:
    table = pd.DataFrame([row for row in results if row["dedupe_mode"] in modes])
    means = table.groupby("dedupe_mode")["total_time"].mean().sort_values()
    best_mode = str(means.index[0])
    return best_mode, float(means.iloc[0])


def mean_total_time_by_method(results: list[dict]) -> pd.Series:
    table = pd.DataFrame(results)
    return table.groupby("dedupe_mode")["total_time"].mean().reindex(DEDUPE_MODES)


def format_duplicate_rate(value: int | None) -> str:
    if value is None:
        return "not_observed"
    return str(value)


def print_key_value_table(title: str, rows: list[tuple[str, str]]) -> None:
    width = max(len(key) for key, _ in rows + [("metric", "value")])
    print(f"\n=== {title} ===")
    print(f"{'metric':<{width}} value")
    for key, value in rows:
        print(f"{key:<{width}} {value}")


def print_final_summary(results: list[dict]) -> None:
    overall_best, overall_time = best_mode_by_mean_time(results, DEDUPE_MODES)
    dedupe_best, dedupe_time = best_mode_by_mean_time(results, DEDUPE_ENABLED_MODES)
    crossover = first_exact_map_crossover(results)
    mean_times = mean_total_time_by_method(results)
    no_dedupe_time = float(mean_times.loc["no_dedupe"])
    duplicate_rates = sorted({row["duplicate_rate"] for row in results})

    print_key_value_table(
        "Decision Metrics",
        [
            ("overall_fastest_method", overall_best),
            ("overall_fastest_mean_total_time", f"{overall_time:.6f}"),
            ("selected_deduplication_method", dedupe_best),
            ("selected_deduplication_mean_total_time", f"{dedupe_time:.6f}"),
            (
                "selected_deduplication_time_gap_vs_no_dedupe",
                f"{dedupe_time - no_dedupe_time:.6f}",
            ),
            (
                "exact_map_runtime_crossover_duplicate_rate",
                format_duplicate_rate(crossover),
            ),
            ("duplicate_rate_sweep", ",".join(str(rate) for rate in duplicate_rates)),
            ("fixed_fingerprint", "packet_counts"),
            ("rf_branch_only", "true"),
        ],
    )

    mean_table = mean_times.reset_index()
    mean_table.columns = ["method", "mean_total_time"]
    print("\n=== Mean Total Time by Method ===")
    print(
        mean_table.to_string(
            index=False,
            formatters={"mean_total_time": "{:.6f}".format},
        )
    )

    crossover_table = pd.DataFrame([{
        "reference_method": "no_dedupe",
        "compared_method": "exact_map",
        "crossover_duplicate_rate": format_duplicate_rate(crossover),
    }])
    print("\n=== Runtime Crossover Check ===")
    print(crossover_table.to_string(index=False))


def main() -> None:
    args = parse_args()

    print("Scope: RF branch only; Snort is not part of this experiment.")
    print("Fingerprint mode: PACKET_COUNTS")
    print(f"Bloom bits: {args.bloom_bits}")
    print(f"Bloom hashes: {args.bloom_hashes}")

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

    detector = RFDetector()
    results = []

    for rate in args.duplicate_rates:
        injected_df, injection_stats = inject_exact_replays(
            df,
            duplicate_rate=rate,
            random_seed=args.duplicate_seed,
            allowed_rates=DUPLICATE_RATES,
        )
        print("\nDuplicate injection:")
        print(f"mode: {injection_stats['injection_mode']}")
        print(f"duplicate_rate: {injection_stats['duplicate_rate']}%")
        print(f"random_seed: {injection_stats['random_seed']}")
        print(f"original_rows: {injection_stats['original_rows']}")
        print(f"injected_rows: {injection_stats['injected_rows']}")
        print(f"total_rows_after_injection: {injection_stats['total_rows']}")

        run_df = add_packet_count_fingerprints(injected_df)
        for mode in DEDUPE_MODES:
            results.append(run_one(run_df, detector, mode, rate, args))

    results = add_no_dedupe_deltas(results)
    print_result_table(results)
    print_final_summary(results)


if __name__ == "__main__":
    main()
