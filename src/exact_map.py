from __future__ import annotations

from collections import deque
import pandas as pd


class ExactMapDedupe:
    def __init__(self, max_recent: int = 50000) -> None:
        self.max_recent = max_recent
        self.seen: set[str] = set()
        self.order: deque[str] = deque()

    def reset(self) -> None:
        self.seen.clear()
        self.order.clear()

    def _insert(self, fp: str) -> None:
        if fp in self.seen:
            return

        self.seen.add(fp)
        self.order.append(fp)

        while len(self.order) > self.max_recent:
            old = self.order.popleft()
            self.seen.discard(old)

    def process_batch(self, batch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        keep_mask: list[bool] = []
        dropped = 0

        for fp in batch["fingerprint"]:
            if fp in self.seen:
                keep_mask.append(False)
                dropped += 1
            else:
                keep_mask.append(True)
                self._insert(fp)

        kept = batch.loc[keep_mask].copy()
        dropped_df = batch.loc[[not x for x in keep_mask]].copy()

        input_events = len(batch)
        output_events = len(kept)

        return kept, dropped_df, {
            "input_events": input_events,
            "output_events": output_events,
            "dropped_duplicates": dropped,
            "drop_rate": dropped / input_events if input_events > 0 else 0.0,
            "state_size": len(self.seen),
        }