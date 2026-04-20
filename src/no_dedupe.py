from __future__ import annotations

import pandas as pd


class NoDedupe:
    def process_batch(
        self,
        batch: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        kept = batch.copy()
        dropped = batch.iloc[0:0].copy()

        return kept, dropped, {
            "input_events": len(batch),
            "output_events": len(batch),
            "dropped_duplicates": 0,
            "drop_rate": 0.0,
            "state_size": 0,
        }


def process_no_dedupe(batch: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    kept, _, stats = NoDedupe().process_batch(batch)
    return kept, stats
