from __future__ import annotations

from collections import deque
import pandas as pd


class ExactMapDedupe:
    def __init__(self, max_recent: int = 50000) -> None:
        self.max_recent = max_recent
        self.seen: set[str] = set()
        self.order: deque[str] = deque()

    def _insert(self, fp: str) -> None:
        if fp in self.seen:
            return

        self.seen.add(fp)
        self.order.append(fp)

        while len(self.order) > self.max_recent:
            old = self.order.popleft()
            self.seen.remove(old)

    def process_batch(self, batch: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        keep_mask = []
        dropped = 0

        for fp in batch["fingerprint"]:
            if fp in self.seen:
                keep_mask.append(False)
                dropped += 1
            else:
                keep_mask.append(True)
                self._insert(fp)

        out = batch.loc[keep_mask].copy()

        return out, {
            "input_events": len(batch),
            "output_events": len(out),
            "dropped_duplicates": dropped,
            "state_size": len(self.seen),
        }