from __future__ import annotations

import numpy as np
import pandas as pd


ALLOWED_DUPLICATE_RATES = [0, 5, 10, 20]


def inject_exact_replays(
    df: pd.DataFrame,
    duplicate_rate: int,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    if duplicate_rate not in ALLOWED_DUPLICATE_RATES:
        raise ValueError(f"duplicate_rate must be one of {ALLOWED_DUPLICATE_RATES}")

    original_rows = len(df)
    duplicate_rows = int(round(original_rows * duplicate_rate / 100))

    stats = {
        "injection_mode": "exact_replay",
        "duplicate_rate": duplicate_rate,
        "random_seed": random_seed,
        "original_rows": original_rows,
        "injected_rows": duplicate_rows,
        "total_rows": original_rows + duplicate_rows,
    }

    if duplicate_rows == 0:
        return df.copy(), stats

    rng = np.random.default_rng(random_seed)
    replay_positions = np.sort(
        rng.choice(original_rows, size=duplicate_rows, replace=False)
    )

    originals = df.copy()
    originals["_replay_order"] = np.arange(original_rows) * 2

    replays = df.iloc[replay_positions].copy()
    replays["_replay_order"] = replay_positions * 2 + 1

    out = (
        pd.concat([originals, replays], ignore_index=True)
        .sort_values("_replay_order", kind="mergesort")
        .drop(columns="_replay_order")
        .reset_index(drop=True)
    )
    return out, stats
