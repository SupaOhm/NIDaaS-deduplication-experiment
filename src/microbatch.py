from __future__ import annotations
from typing import Iterator
import pandas as pd

def iter_microbatches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size].copy()