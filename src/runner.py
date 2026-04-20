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
from src.no_dedupe import NoDedupe
from src.exact_map import ExactMapDedupe
from src.bloom import BloomDedupe
from src.bloom_exact import BloomExactDedupe


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
DEFAULT_DEDUPE_MODE = "exact_map"
DEDUPE_MODES = ["no_dedupe", DEFAULT_DEDUPE_MODE, "bloom", "bloom_exact"]
DEDUPE_MODE_GROUPS = {
    "all": DEDUPE_MODES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dedupe + RF experiment.")
    parser.add_argument(
        "--fingerprint-mode",
        choices=[*FINGERPRINT_BUILDERS, *FINGERPRINT_MODE_GROUPS],
        default=DEFAULT_FINGERPRINT_MODE,
        help=(
            f"Fingerprint variant to test. Default is {DEFAULT_FINGERPRINT_MODE}. "
            "Use 'ab' for basic vs duration, or 'all' for all variants."
        ),
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=[*DEDUPE_MODES, *DEDUPE_MODE_GROUPS],
        default=DEFAULT_DEDUPE_MODE,
        help=(
            f"Dedupe strategy to test. Default is {DEFAULT_DEDUPE_MODE}. "
            "Use 'all' to compare NoDedupe, ExactMap, Bloom, and Bloom+ExactMap."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-recent", type=int, default=50000)
    parser.add_argument("--bloom-bits", type=int, default=50_000_000)
    parser.add_argument("--bloom-hashes", type=int, default=4)
    return parser.parse_args()


def selected_fingerprint_modes(mode: str) -> list[str]:
    return FINGERPRINT_MODE_GROUPS.get(mode, [mode])


def selected_dedupe_modes(mode: str) -> list[str]:
    return DEDUPE_MODE_GROUPS.get(mode, [mode])


def add_fingerprints(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["fingerprint"] = out.apply(FINGERPRINT_BUILDERS[mode], axis=1)
    return out


def make_deduper(args: argparse.Namespace, mode: str):
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


def run_experiment(
    df: pd.DataFrame,
    fingerprint_mode: str,
    dedupe_mode: str,
    args: argparse.Namespace,
) -> dict:
    print(f"\nFingerprint mode: {fingerprint_mode.upper()}")
    print(f"Dedupe mode: {dedupe_mode.upper()}")

    metrics = RunMetrics()
    detector = RFDetector()
    deduper = make_deduper(args, dedupe_mode)
    last_stats: dict = {"state_size": 0}

    y_true_all = []
    y_pred_all = []

    dropped_benign = 0
    dropped_attack = 0

    for batch in iter_microbatches(df, args.batch_size):
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
        "fingerprint_mode": fingerprint_mode,
        "dedupe_mode": dedupe_mode,
        "drop_rate": drop_rate,
        "final_state_size": last_stats["state_size"],
        "dropped_benign": dropped_benign,
        "dropped_attack": dropped_attack,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    for key in [
        "bit_size",
        "num_hashes",
        "inserted_count",
        "bloom_bits_set",
        "bloom_inserted",
        "bloom_maybe_seen",
        "bloom_false_positives",
    ]:
        if key in last_stats:
            result[key] = last_stats[key]

    print(f"\n=== {dedupe_mode.upper()} + RF Baseline ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"final_state_size: {result['final_state_size']}")
    if dedupe_mode in {"bloom", "bloom_exact"}:
        print(f"bloom_bits: {result['bit_size']}")
        print(f"bloom_hashes: {result['num_hashes']}")
    if dedupe_mode == "bloom":
        print(f"bloom_inserted: {result['inserted_count']}")
    if dedupe_mode == "bloom_exact":
        print(f"bloom_bits_set: {result['bloom_bits_set']}")
        print(f"bloom_inserted: {result['bloom_inserted']}")
        print(f"bloom_maybe_seen: {result['bloom_maybe_seen']}")
        print(f"bloom_false_positives: {result['bloom_false_positives']}")
    print(f"dropped_benign: {dropped_benign}")
    print(f"dropped_attack: {dropped_attack}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")

    return result


def print_comparison_table(results: list[dict]) -> None:
    if len(results) < 2:
        return

    print("\n=== Experiment Comparison ===")
    print(
        f"{'dedupe':<12} {'fingerprint':<16} {'dropped':>10} {'drop_rate':>10} "
        f"{'drop_attack':>12} {'drop_benign':>12} "
        f"{'precision':>10} {'recall':>8} {'f1':>8}"
    )
    for result in results:
        print(
            f"{result['dedupe_mode']:<12} "
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
    if {result["dedupe_mode"] for result in results} != {"exact_map"}:
        return
    if [result["fingerprint_mode"] for result in results] != ["basic", "duration"]:
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

    fingerprint_modes = selected_fingerprint_modes(args.fingerprint_mode)
    dedupe_modes = selected_dedupe_modes(args.dedupe_mode)
    results = []

    for fingerprint_mode in fingerprint_modes:
        run_df = add_fingerprints(df, fingerprint_mode)
        for dedupe_mode in dedupe_modes:
            results.append(run_experiment(run_df, fingerprint_mode, dedupe_mode, args))

    print_comparison_table(results)
    print_ab_delta(results)


if __name__ == "__main__":
    main()
