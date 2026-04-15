from __future__ import annotations
import pandas as pd

def process_no_dedupe(batch: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return batch.copy(), {
        "input_events": len(batch),
        "output_events": len(batch),
        "dropped_duplicates": 0,
    }